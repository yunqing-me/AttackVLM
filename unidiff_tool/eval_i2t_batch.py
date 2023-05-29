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
import os
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

        
def i2t_image_batch(config, nnet, use_caption_decoder, caption_decoder, clip_text_model, autoencoder, clip_img_model, clip_img_model_preprocess, image_batch):
    # if config.get('benchmark', False):

    set_seed(config.seed)

    # config = ml_collections.FrozenConfigDict(config)
    utils.set_logger(log_level='info')

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    ##### START loading unidiffuser models
    # done in another file
    ######## END of loading unidiffuser models
    
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


    def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
        """
        1. calculate the conditional model output
        2. calculate unconditional model output
            config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
            config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
        3. return linear combination of conditional output and unconditional output
        """
        z, clip_img = split(x)

        t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)

        if config.sample.scale == 0.:
            return x_out

        if config.sample.t2i_cfg_mode == 'empty_token':
            _empty_context = einops.repeat(empty_context, 'L D -> B L D', B=x.size(0))
            if use_caption_decoder:
                _empty_context = caption_decoder.encode_prefix(_empty_context)
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=_empty_context, t_img=timesteps, t_text=t_text,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        elif config.sample.t2i_cfg_mode == 'true_uncond':
            text_N = torch.randn_like(text)  # 3 other possible choices
            z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                                      data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
            x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + config.sample.scale * (x_out - x_out_uncond)


    def i_nnet(x, timesteps):
        z, clip_img = split(x)
        text = torch.randn(x.size(0), 77, config.text_dim, device=device)
        t_text = torch.ones_like(timesteps) * N
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                                             data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + config.data_type)
        x_out = combine(z_out, clip_img_out)
        return x_out

    def t_nnet(x, timesteps):
        z = torch.randn(x.size(0), *config.z_shape, device=device)
        clip_img = torch.randn(x.size(0), 1, config.clip_img_dim, device=device)
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                             data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        return text_out

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

    def split_joint(x):
        C, H, W = config.z_shape
        z_dim = C * H * W
        z, clip_img, text = x.split([z_dim, config.clip_img_dim, 77 * config.text_dim], dim=1)
        z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
        clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=config.clip_img_dim)
        text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=config.text_dim)
        return z, clip_img, text

    def combine_joint(z, clip_img, text):
        z = einops.rearrange(z, 'B C H W -> B (C H W)')
        clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
        text = einops.rearrange(text, 'B L D -> B (L D)')
        return torch.concat([z, clip_img, text], dim=-1)

    def joint_nnet(x, timesteps):
        z, clip_img, text = split_joint(x)
        z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=timesteps,
                                             data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        x_out = combine_joint(z_out, clip_img_out, text_out)

        if config.sample.scale == 0.:
            return x_out

        z_noise = torch.randn(x.size(0), *config.z_shape, device=device)
        clip_img_noise = torch.randn(x.size(0), 1, config.clip_img_dim, device=device)
        text_noise = torch.randn(x.size(0), 77, config.text_dim, device=device)

        _, _, text_out_uncond = nnet(z_noise, clip_img_noise, text=text, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                     data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)
        z_out_uncond, clip_img_out_uncond, _ = nnet(z, clip_img, text=text_noise, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                                    data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + config.data_type)

        x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

        return x_out + config.sample.scale * (x_out - x_out_uncond)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)


    logging.info(config.sample)
    logging.info(f'N={N}')

    # use a batch of images to obtain results
    #####
    # get contexts
    resolution  = config.z_shape[-1] * 8
    image_batch = image_batch.cpu()
    image = einops.rearrange(image_batch, 'n c h w -> n h w c').contiguous().numpy().astype(np.uint8)  # (num_query*batch_size, h, w, c)
    
    with torch.no_grad():
        # clip_img_feature = clip_img_model.encode_image(clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device))

        center_cropped_img = []
        clip_processed_img = []
        for idx in range(image.shape[0]):
            center_cropped_img.append(utils.center_crop(resolution, resolution, image[idx]))
            clip_processed_img.append(clip_img_model_preprocess(Image.fromarray(image[idx])).to(device))
        
        clip_processed_img = torch.stack(clip_processed_img, dim=0)
        clip_img_feature = clip_img_model.encode_image(clip_processed_img).unsqueeze(1)  # size = (num_query*batch_size, 512)
        image = np.stack(center_cropped_img, axis=0)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'n h w c -> n c h w')
        image = torch.tensor(image, device=device)
        moments = autoencoder.encode_moments(image)

    clip_img, img_context = clip_img_feature, moments
    contexts = torch.randn(image.shape[0], 77, config.clip_text_dim).to(device)
    contexts = contexts  # the clip embedding of conditioned texts

    print("context size:", contexts.size())
    print("img_contexts size:", img_context.squeeze(1).size())
    print("clip_imgs size:", clip_img.size())
    #####
    
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
        if mode == 'joint':
            _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
        elif mode in ['t2i', 'i']:
            _x_init = combine(_z_init, _clip_img_init)
        elif mode in ['i2t', 't']:
            _x_init = _text_init
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * N
            if mode == 'joint':
                return joint_nnet(x, t)
            elif mode == 't2i':
                return t2i_nnet(x, t, **kwargs)
            elif mode == 'i2t':
                return i2t_nnet(x, t, **kwargs)
            elif mode == 'i':
                return i_nnet(x, t)
            elif mode == 't':
                return t_nnet(x, t)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type=device):
                start_time = time.time()
                x = dpm_solver.sample(_x_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
                end_time = time.time()
                print(f'\ngenerate {_n_samples} samples with {config.sample.sample_steps} steps takes {end_time - start_time:.2f}s')

        if mode == 'joint':
            _z, _clip_img, _text = split_joint(x)
            return _z, _clip_img, _text
        elif mode in ['t2i', 'i']:
            _z, _clip_img = split(x)
            return _z, _clip_img
        elif mode in ['i2t', 't']:
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

    return samples
