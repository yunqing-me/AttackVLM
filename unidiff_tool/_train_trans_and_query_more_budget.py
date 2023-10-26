import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from eval_i2t_batch import i2t_image_batch
import wandb

### unidiff lib
import ml_collections
import utils
from absl import logging
import libs.autoencoder
import libs.clip
### ###########

# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  

def d(**kwconfig):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwconfig)

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
    ]
)

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)

def find_max(arr):
    max_value = float('-inf')
    for num in arr:
        if num > max_value:
            max_value = num
    return max_value


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnet_path", default="models/uvit_v1.pth", type=str)
    parser.add_argument("--mode", default="i2t", type=str)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--output", default="temp", type=str)
    parser.add_argument("--data_path", default="temp", type=str)
    parser.add_argument("--text_path", default="temp", type=str)
    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--save_img", action='store_true')
    parser.add_argument("--num_query", default=5, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=16, type=float)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    
    config = parser.parse_args()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.text_dim
    )

    config.nnet = d(
        name='uvit_multi_post_ln_v1',
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.text_dim,
        num_text_tokens=77,
        clip_img_dim=config.clip_img_dim,
        use_checkpoint=True
    )
    config.sample = d(
        sample_steps=50,
        scale=7.,
        t2i_cfg_mode='true_uncond'
    )
    
    # ---------------------- #
    print("Loading unidiffuser models..")
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

    # text model for unidiffuser
    clip_text_model_for_unidiff = libs.clip.FrozenCLIPEmbedder(device=device)
    clip_text_model_for_unidiff = clip_text_model_for_unidiff.to(device)
    clip_text_model_for_unidiff.eval()
    
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    # use clip text encoder for attack
    clip_img_model_for_unidiff, clip_img_model_preprocess_for_unidiff = clip.load("ViT-B/32", device=device, jit=False)
    
    clip_img_model_rn50,   _ = clip.load("RN50", device=device, jit=False)
    clip_img_model_rn101,  _ = clip.load("RN101", device=device, jit=False)
    clip_img_model_vitb16, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_img_model_vitl14, _ = clip.load("ViT-L/14", device=device, jit=False)
    print("Done")
    # ---------------------- #

    # load clip_model params
    num_sub_query, num_query, sigma = config.num_sub_query, config.num_query, config.sigma
    batch_size    = config.batch_size
    alpha         = config.alpha
    epsilon       = config.epsilon
    vit_adv_data  = ImageFolderWithPaths(config.data_path, transform=transform)
    data_loader   = torch.utils.data.DataLoader(vit_adv_data, batch_size=batch_size, shuffle=False, num_workers=24)
    
    # org text/features
    adv_vit_text_path = config.text_path
    with open(os.path.join(adv_vit_text_path), 'r') as f:
        unidiff_text_of_adv_vit  = f.readlines()[:config.num_samples] 
        f.close()
    
    with torch.no_grad():
        adv_vit_text_token   = clip.tokenize(unidiff_text_of_adv_vit).to(device)
        adv_vit_text_features = clip_img_model_for_unidiff.encode_text(adv_vit_text_token)
        adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
        adv_vit_text_features = adv_vit_text_features.detach()
    
    # tgt text/features
    tgt_text_path = './_coco_captions_10000.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:config.num_samples] 
        f.close()
    
    # clip text features of the target
    with torch.no_grad():
        target_text_token    = clip.tokenize(tgt_text).to(device)
        target_text_features = clip_img_model_for_unidiff.encode_text(target_text_token)
        target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
        target_text_features = target_text_features.detach()

    # baseline results
    vit_attack_results   = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    query_attack_results = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    assert (vit_attack_results == query_attack_results).all()
    
    ## other arch
    with torch.no_grad():
        # rn50
        adv_vit_text_features_rn50 = clip_img_model_rn50.encode_text(adv_vit_text_token)
        adv_vit_text_features_rn50 = adv_vit_text_features_rn50 / adv_vit_text_features_rn50.norm(dim=1, keepdim=True)
        adv_vit_text_features_rn50 = adv_vit_text_features_rn50.detach()
        target_text_features_rn50  = clip_img_model_rn50.encode_text(target_text_token)
        target_text_features_rn50  = target_text_features_rn50 / target_text_features_rn50.norm(dim=1, keepdim=True)
        target_text_features_rn50  = target_text_features_rn50.detach()
        vit_attack_results_rn50    = torch.sum(adv_vit_text_features_rn50 * target_text_features_rn50, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_rn50  = torch.sum(adv_vit_text_features_rn50 * target_text_features_rn50, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_rn50 == query_attack_results_rn50).all()

        # rn101
        adv_vit_text_features_rn101 = clip_img_model_rn101.encode_text(adv_vit_text_token)
        adv_vit_text_features_rn101 = adv_vit_text_features_rn101 / adv_vit_text_features_rn101.norm(dim=1, keepdim=True)
        adv_vit_text_features_rn101 = adv_vit_text_features_rn101.detach()
        target_text_features_rn101  = clip_img_model_rn101.encode_text(target_text_token)
        target_text_features_rn101  = target_text_features_rn101 / target_text_features_rn101.norm(dim=1, keepdim=True)
        target_text_features_rn101  = target_text_features_rn101.detach()
        vit_attack_results_rn101    = torch.sum(adv_vit_text_features_rn101 * target_text_features_rn101, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_rn101  = torch.sum(adv_vit_text_features_rn101 * target_text_features_rn101, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_rn101 == query_attack_results_rn101).all()

        # vitb16
        adv_vit_text_features_vitb16 = clip_img_model_vitb16.encode_text(adv_vit_text_token)
        adv_vit_text_features_vitb16 = adv_vit_text_features_vitb16 / adv_vit_text_features_vitb16.norm(dim=1, keepdim=True)
        adv_vit_text_features_vitb16 = adv_vit_text_features_vitb16.detach()
        target_text_features_vitb16  = clip_img_model_vitb16.encode_text(target_text_token)
        target_text_features_vitb16  = target_text_features_vitb16 / target_text_features_vitb16.norm(dim=1, keepdim=True)
        target_text_features_vitb16  = target_text_features_vitb16.detach()
        vit_attack_results_vitb16    = torch.sum(adv_vit_text_features_vitb16 * target_text_features_vitb16, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_vitb16  = torch.sum(adv_vit_text_features_vitb16 * target_text_features_vitb16, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_vitb16 == query_attack_results_vitb16).all()

        # vitl14
        adv_vit_text_features_vitl14 = clip_img_model_vitl14.encode_text(adv_vit_text_token)
        adv_vit_text_features_vitl14 = adv_vit_text_features_vitl14 / adv_vit_text_features_vitl14.norm(dim=1, keepdim=True)
        adv_vit_text_features_vitl14 = adv_vit_text_features_vitl14.detach()
        target_text_features_vitl14  = clip_img_model_vitl14.encode_text(target_text_token)
        target_text_features_vitl14  = target_text_features_vitl14 / target_text_features_vitl14.norm(dim=1, keepdim=True)
        target_text_features_vitl14  = target_text_features_vitl14.detach()
        vit_attack_results_vitl14    = torch.sum(adv_vit_text_features_vitl14 * target_text_features_vitl14, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_vitl14  = torch.sum(adv_vit_text_features_vitl14 * target_text_features_vitl14, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_vitl14 == query_attack_results_vitl14).all()
    ## ----------
    
    if config.wandb:
        run = wandb.init(project=config.wandb_project_name, name=config.wandb_run_name, reinit=True)
    
    for i, (image, label, path) in enumerate(data_loader):
        if batch_size * (i+1) > config.num_samples:
            break
        image = image.to(device)  # size=(10, 3, 224, 224)
        
        # obtain all text features (via CLIP text encoder)
        adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)]        
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)]
        
        # ------------------- random gradient-free method
        if config.delta == 'normal':
            delta = torch.randn_like(image, requires_grad=False)
        elif config.delta == 'zero':
            delta = torch.zeros_like(image, requires_grad=False)
        torch.cuda.empty_cache()
        
        best_caption = unidiff_text_of_adv_vit[i]
        better_flag = 0
        for step_idx in range(config.steps):
            print(f"{i}-th image / {step_idx}-th step")
            # step 1. obtain purturbed images
            
            ##########
            ##########
            with torch.no_grad():
                if step_idx == 0:
                    image_repeat      = image.repeat(num_query, 1, 1, 1)
                else:
                    image_repeat      = adv_image_in_current_step.repeat(num_query, 1, 1, 1)             

                    unidiff_text_of_adv_vit_in_current_step = i2t_image_batch(config, nnet, use_caption_decoder, caption_decoder, clip_text_model_for_unidiff, autoencoder,
                                                                    clip_img_model_for_unidiff, clip_img_model_preprocess_for_unidiff, adv_image_in_current_step)
                    adv_vit_text_token_in_current_step      = clip.tokenize(unidiff_text_of_adv_vit_in_current_step).to(device)
                    adv_vit_text_features_in_current_step   = clip_img_model_for_unidiff.encode_text(adv_vit_text_token_in_current_step)
                    adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
                    adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step.detach()                
                    adv_text_features     = adv_vit_text_features_in_current_step
                    torch.cuda.empty_cache()
            ###########
            ###########
            
            query_noise            = torch.randn_like(image_repeat).sign() # Rademacher noise
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, 224, 224)
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []
            for query_idx in range(num_query//num_sub_query):
                sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
                with torch.no_grad():
                    text_of_sub_perturbed_imgs = i2t_image_batch(config, nnet, use_caption_decoder, caption_decoder, clip_text_model_for_unidiff, autoencoder,
                                                                 clip_img_model_for_unidiff, clip_img_model_preprocess_for_unidiff,
                                                                 sub_perturbed_image_repeat)  # a list with length num_sub_query
                text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs)
            
            # step 2. estimate grad
            with torch.no_grad():
                perturb_text_token    = clip.tokenize(text_of_perturbed_imgs).to(device)
                perturb_text_features = clip_img_model_for_unidiff.encode_text(perturb_text_token)
                perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
                perturb_text_features = perturb_text_features.detach()
            
            coefficient     = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)  # size = (num_query * batch_size)
            coefficient     = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise     = query_noise.reshape(num_query, batch_size, 3, 224, 224)
            pseudo_gradient = coefficient * query_noise / sigma # size = (num_query, batch_size, 3, 224, 224)
            pseudo_gradient = pseudo_gradient.mean(0) # size = (bs, 3, 224, 224)
            
            # step 3. log metrics
            with torch.no_grad():

                delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
                delta.data = delta_data
                adv_image_in_current_step = torch.clamp(image+delta, 0.0, 255.0)
                
                print(f"{i}-th img // {step_idx}-th step // max  delta", torch.max(torch.abs(delta)).item())
                print(f"{i}-th img // {step_idx}-th step // mean delta", torch.mean(torch.abs(delta)).item())
                
                # wandb.log(
                #     {
                #         "max delta:":  torch.max(torch.abs(delta)).item(),
                #         "mean delta:": torch.mean(torch.abs(delta)).item()
                #     }
                # )
                unidiff_text_of_adv_image_in_current_step = i2t_image_batch(config, nnet, use_caption_decoder, caption_decoder, clip_text_model_for_unidiff, autoencoder,
                                                                            clip_img_model_for_unidiff, clip_img_model_preprocess_for_unidiff, adv_image_in_current_step)

                unidiff_text_token = clip.tokenize(unidiff_text_of_adv_image_in_current_step).to(device)
                unidiff_text_features_of_adv_image_in_current_step = clip_img_model_for_unidiff.encode_text(unidiff_text_token)
                unidiff_text_features_of_adv_image_in_current_step = unidiff_text_features_of_adv_image_in_current_step / unidiff_text_features_of_adv_image_in_current_step.norm(dim=1, keepdim=True)
                unidiff_text_features_of_adv_image_in_current_step = unidiff_text_features_of_adv_image_in_current_step.detach()

                adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(unidiff_text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
                
                # update results
                if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
                    query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
                    best_caption = unidiff_text_of_adv_image_in_current_step[0]
                    better_flag  = 1
            
                # other clip archs
                # rn50
                tgt_text_features_rn50 = target_text_features_rn50[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn50 = clip_img_model_rn50.encode_text(unidiff_text_token)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50 / text_features_of_adv_image_in_current_step_rn50.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50.detach()
                adv_txt_tgt_txt_score_in_current_step_rn50 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn50 * tgt_text_features_rn50, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn50 > query_attack_results_rn50[i]:
                    query_attack_results_rn50[i] = adv_txt_tgt_txt_score_in_current_step_rn50
                
                # rn101
                tgt_text_features_rn101 = target_text_features_rn101[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn101 = clip_img_model_rn101.encode_text(unidiff_text_token)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101 / text_features_of_adv_image_in_current_step_rn101.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101.detach()
                adv_txt_tgt_txt_score_in_current_step_rn101 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn101 * tgt_text_features_rn101, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn101 > query_attack_results_rn101[i]:
                    query_attack_results_rn101[i] = adv_txt_tgt_txt_score_in_current_step_rn101
                
                # vitb16
                tgt_text_features_vitb16 = target_text_features_vitb16[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitb16 = clip_img_model_vitb16.encode_text(unidiff_text_token)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16 / text_features_of_adv_image_in_current_step_vitb16.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16.detach()
                adv_txt_tgt_txt_score_in_current_step_vitb16 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitb16 * tgt_text_features_vitb16, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitb16 > query_attack_results_vitb16[i]:
                    query_attack_results_vitb16[i] = adv_txt_tgt_txt_score_in_current_step_vitb16
                
                # vitl14
                tgt_text_features_vitl14 = target_text_features_vitl14[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitl14 = clip_img_model_vitl14.encode_text(unidiff_text_token)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14 / text_features_of_adv_image_in_current_step_vitl14.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14.detach()
                adv_txt_tgt_txt_score_in_current_step_vitl14 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitl14 * tgt_text_features_vitl14, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitl14 > query_attack_results_vitl14[i]:
                    query_attack_results_vitl14[i] = adv_txt_tgt_txt_score_in_current_step_vitl14
                    # ----------------
                torch.cuda.empty_cache()
                
        # log clip score after query
        if config.wandb:
            wandb.log(
                {   
                    "moving-avg-adv-rn50"    : np.mean(vit_attack_results_rn50[:(i+1)]),
                    "moving-avg-query-rn50"  : np.mean(query_attack_results_rn50[:(i+1)]),
                    
                    "moving-avg-adv-rn101"   : np.mean(vit_attack_results_rn101[:(i+1)]),
                    "moving-avg-query-rn101" : np.mean(query_attack_results_rn101[:(i+1)]),
                    
                    "moving-avg-adv-vitb16"  : np.mean(vit_attack_results_vitb16[:(i+1)]),
                    "moving-avg-query-vitb16": np.mean(query_attack_results_vitb16[:(i+1)]),
                    
                    "moving-avg-adv-vitb32"  : np.mean(vit_attack_results[:(i+1)]),
                    "moving-avg-query-vitb32": np.mean(query_attack_results[:(i+1)]),
                    
                    "moving-avg-adv-vitl14"  : np.mean(vit_attack_results_vitl14[:(i+1)]),
                    "moving-avg-query-vitl14": np.mean(query_attack_results_vitl14[:(i+1)]),
                }
            )

        # log text after query
        print("best caption of current image:", best_caption)
        with open(os.path.join(config.output + '.txt'), 'a') as f:
            if better_flag:
                f.write(best_caption+'\n')
            else:
                f.write(best_caption)
        f.close()