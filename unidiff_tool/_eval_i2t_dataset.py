import ml_collections
import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
import clip
from PIL import Image
import time

import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def prepare_contexts(config, clip_text_model, clip_img_model, clip_img_model_preprocess, autoencoder):
    resolution = config.z_shape[-1] * 8

    contexts = torch.randn(config.n_samples, 77, config.clip_text_dim).to(device)
    img_contexts = torch.randn(config.n_samples, 2 * config.z_shape[0], config.z_shape[1], config.z_shape[2])
    clip_imgs = torch.randn(config.n_samples, 1, config.clip_img_dim)

    if config.mode in ['t2i', 't2i2t']:
        prompts = [ config.prompt ] * config.n_samples
        contexts = clip_text_model.encode(prompts)

    elif config.mode in ['i2t', 'i2t2i']:
        from PIL import Image
        img_contexts = []
        clip_imgs = []

        def get_img_feature(image):
            image = np.array(image).astype(np.uint8)
            image = utils.center_crop(resolution, resolution, image)
            c_image = clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            # print("clip image shape:", c_image.size())
            clip_img_feature = clip_img_model.encode_image(c_image)

            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, 'h w c -> 1 c h w')
            image = torch.tensor(image, device=device)
            moments = autoencoder.encode_moments(image)

            return clip_img_feature, moments

        image = Image.open(config.img).convert('RGB')
        clip_img, img_context = get_img_feature(image)
        img_contexts.append(img_context)
        clip_imgs.append(clip_img)
        img_contexts = img_contexts * config.n_samples
        clip_imgs = clip_imgs * config.n_samples
        
        img_contexts = torch.concat(img_contexts, dim=0)
        clip_imgs = torch.stack(clip_imgs, dim=0)

    return contexts, img_contexts, clip_imgs


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class ImageFolderForUniDiff(torchvision.datasets.ImageFolder):
    def __init__(self, root, resolution, clip_img_model, clip_img_model_preprocess, autoencoder, transform = None, target_transform = None, loader = ..., is_valid_file = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.resolution = resolution
        self.clip_img_model_preprocess = clip_img_model_preprocess
        self.clip_img_model = clip_img_model
        self.autoencoder = autoencoder
    
    def __getitem__(self, index: int):
        # original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        image = Image.open(path).convert('RGB')
        # clip_img, img_context = get_img_feature(image)
   
        image = np.array(image).astype(np.uint8)
        image = utils.center_crop(self.resolution, self.resolution, image)
        
        with torch.no_grad():
            clip_img_feature = self.clip_img_model.encode_image(self.clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device))

            image = (image / 127.5 - 1.0).astype(np.float32)
            image = einops.rearrange(image, 'h w c -> 1 c h w')
            image = torch.tensor(image, device=device)
            moments = self.autoencoder.encode_moments(image)

        return (clip_img_feature, moments, path)
        
        
def evaluate(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    set_seed(config.seed)

    config = ml_collections.FrozenConfigDict(config)
    utils.set_logger(log_level='info')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    nnet = utils.get_nnet(**config.nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.to(device)
    nnet.eval()

    use_caption_decoder = config.text_dim < config.clip_text_dim or config.mode != 't2i'
    if use_caption_decoder:
        from libs.caption_decoder import CaptionDecoder
        caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    else:
        caption_decoder = None

    clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model.eval()
    clip_text_model.to(device)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    empty_context = clip_text_model.encode([''])[0]

    def split(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img = x.split([z_dim, config.clip_img_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        return z, clip_img


    def combine(z, clip_img):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        return torch.concat([z, clip_img], dim=-1)


    def i2t_nnet(x, timesteps, z, clip_img):
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
        3. return linear combination of conditional output and unconditional output
        """
        t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=x, t_img=t_img, t_text=timesteps,
                                             data_type=torch.zeros_like(t_img, device=device, dtype=torch.int) + config.data_type)

        if config.sample.scale == 0.:
            return text_out

        z_N = torch.randn_like(z)  # 3 other possible choices
        clip_img_N = torch.randn_like(clip_img)
        z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z_N, clip_img_N, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                                                  data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)

        return text_out + config.sample.scale * (text_out - text_out_uncond)

    logging.info(config.sample)
    logging.info(f'N={N}')

    context_dataset = ImageFolderForUniDiff(config.img_path, resolution=config.z_shape[-1] * 8,
                                            clip_img_model=clip_img_model, clip_img_model_preprocess=clip_img_model_preprocess, autoencoder=autoencoder, transform=None)
    context_dataloader = torch.utils.data.DataLoader(context_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8)
    
    for i, (clip_img, img_context, path) in enumerate(context_dataloader):
        
        contexts = torch.randn(config.batch_size, 77, config.clip_text_dim).to(device)
        
        contexts = contexts  # the clip embedding of conditioned texts
        contexts_low_dim = contexts if not use_caption_decoder else caption_decoder.encode_prefix(contexts)  # the low dimensional version of the contexts, which is the input to the nnet

        img_contexts = img_context.squeeze(1)  # img_contexts is the autoencoder moment
        z_img = autoencoder.sample(img_contexts)
        clip_imgs = clip_img  # the clip embedding of conditioned image

        if config.mode in ['t2i', 't2i2t']:
            _n_samples = contexts_low_dim.size(0)
        elif config.mode in ['i2t', 'i2t2i']:
            _n_samples = img_contexts.size(0)
        else:
            _n_samples = config.n_samples


        def sample_fn(mode, **kwargs):

            _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
            _clip_img_init = torch.randn(_n_samples, 1, config.clip_img_dim, device=device)
            _text_init = torch.randn(_n_samples, 77, config.text_dim, device=device)
            if mode in ['i2t', 't']:
                _x_init = _text_init
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

            def model_fn(x, t_continuous):
                t = t_continuous * N
                if mode == 'i2t':
                    return i2t_nnet(x, t, **kwargs)

            dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
            with torch.no_grad():
                with torch.autocast(device_type=device):
                    start_time = time.time()
                    x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                    end_time = time.time()
                    print(f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

            os.makedirs(os.path.join("output_unidiffuser", config.output), exist_ok=True)
            if mode in ['i2t', 't']:
                return x

        if config.mode in ['i2t', 't', 't2i2t']:
            if config.mode == 'i2t':
                _text = sample_fn(config.mode, z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
            elif config.mode == 't':
                _text = sample_fn(config.mode)
            elif config.mode == 't2i2t':
                _z, _clip_img = sample_fn('t2i', text=contexts_low_dim)
                _text = sample_fn('i2t', z=_z, clip_img=_clip_img)
            samples = caption_decoder.generate_captions(_text)
            logging.info(samples)
            
            os.makedirs(os.path.join("output_unidiffuser", config.output), exist_ok=True)
            
            with open(os.path.join("output_unidiffuser", f'{config.output}.txt'), 'a') as f:
                print('\n'.join(samples), file=f)
            f.close()

        print(f'\nGPU memory usage: {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB')
        print(f'\nresults are saved in {os.path.join("output_unidiffuser", config.output)} :)')


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "configs/sample_unidiffuser_v1.py", "Configuration.", lock_config=False)
flags.DEFINE_string("nnet_path", "models/uvit_v1.pth", "The nnet to evaluate.")
flags.DEFINE_string("output", "out", "dir to write results to")
flags.DEFINE_string("prompt", "an elephant under the sea", "the prompt for text-to-image generation and text variation")
flags.DEFINE_string("img", "assets/space.jpg", "the image path for image-to-text generation and image variation")
flags.DEFINE_integer("n_samples", 1, "the number of samples to generate")

flags.DEFINE_string("img_path", "output/clip_adv_targeted_text/", "dir to conditioning imgs")
flags.DEFINE_integer("batch_size", 8, "the number of samples sampled from existing dataloader")

flags.DEFINE_integer("nrow", 4, "number of images displayed in each row of the grid")
flags.DEFINE_string("mode", None,
                    "type of generation, one of t2i / i2t / joint / i / t / i2t2i/ t2i2t\n"
                    "t2i: text to image\n"
                    "i2t: image to text\n"
                    "joint: joint generation of text and image\n"
                    "i: only generate image\n"
                    "t: only generate text\n"
                    "i2t2i: image variation, first image to text, then text to image\n"
                    "t2i2t: text variation, first text to image, the image to text\n"
                    )


def main(argv=None):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output = FLAGS.output
    config.prompt = FLAGS.prompt
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.img = FLAGS.img
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode
    
    config.batch_size = FLAGS.batch_size
    config.img_path = FLAGS.img_path
    
    evaluate(config)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    app.run(main)