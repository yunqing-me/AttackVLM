import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
import wandb
import copy

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


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
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        image_processed = vis_processor(original_tuple[0])
        return image_processed, original_tuple[1], path

def _i2t(args, chat, image_tensor):
    
    img_list   = chat.get_img_list(image_tensor, img_list=[])  # img embeddings, size() = [bs, 32, 5120]
    mixed_embs = chat.get_mixed_embs(args, img_list=img_list, caption_size=image_tensor.size()[0])
    captions   = chat.get_text(args, mixed_embs, text_size=image_tensor.size()[0])
    return captions


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    # load models for i2t
    # minigpt-4
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--output", default="tmp", type=str)
    parser.add_argument("--data_path", default="../_output_img/minigpt4_adv", type=str)
    parser.add_argument("--text_path", default="../_output_text/minigpt4_adv_pred.txt", type=str)
    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--num_query", default=20, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=8, type=float)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='tmp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='tmp_run')
    
    args = parser.parse_args()

    # ---------------------- #
    print(f"Loading MiniGPT-4 models...")
    # load models for i2t
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor     = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)       
    num_beams = 1
    temperature = 1.0
    print("Done")
     
    # use clip text coder for attack
    clip_img_model_rn50,   _ = clip.load("RN50", device=device, jit=False)
    clip_img_model_rn101,  _ = clip.load("RN101", device=device, jit=False)
    clip_img_model_vitb16, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_img_model_vitb32, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_img_model_vitl14, _ = clip.load("ViT-L/14", device=device, jit=False)
    # ---------------------- #

    # load clip_model params
    num_sub_query, num_query, sigma = args.num_sub_query, args.num_query, args.sigma
    batch_size    = copy.deepcopy(args.batch_size)
    alpha         = args.alpha
    epsilon       = args.epsilon

    # load image
    adv_vit_data = ImageFolderWithPaths(args.data_path, transform=None)
    dataloader    = torch.utils.data.DataLoader(adv_vit_data, batch_size=batch_size, shuffle=False, num_workers=24)

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))     
    
    # org text/features
    adv_vit_text_path = args.text_path
    with open(os.path.join(adv_vit_text_path), 'r') as f:
        adv_vit_text  = f.readlines()[:args.num_samples] 
        f.close()
    
    with torch.no_grad():
        adv_vit_text_token    = clip.tokenize(adv_vit_text, truncate=True).to(device)
        adv_vit_text_features = clip_img_model_vitb32.encode_text(adv_vit_text_token)
        adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
        adv_vit_text_features = adv_vit_text_features.detach()
    
    # tgt text/features
    tgt_text_path = '../_output_text/_coco_captions_10000.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:args.num_samples] 
        f.close()
    
    # clip text features of the target
    with torch.no_grad():
        target_text_token    = clip.tokenize(tgt_text, truncate=True).to(device)
        target_text_features = clip_img_model_vitb32.encode_text(target_text_token)
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
    
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)
    
    for i, (image, _, path) in enumerate(dataloader):
        if batch_size * (i+1) > args.num_samples:
            break
        image = image.to(device)  # size=(10, 3, args.input_res, args.input_res)
        # obtain all text features (via CLIP text encoder)
        adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)]        
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)]
        
        # ------------------- random gradient-free method
        if args.delta == 'normal':
            delta = torch.randn_like(image, requires_grad=False)
        elif args.delta == 'zero':
            delta = torch.zeros_like(image, requires_grad=False)
        
        for step_idx in range(args.steps):
            print(f"{i}-th image - {step_idx}-th step")
            # step 1. obtain purturbed images
            image_repeat           = image.repeat(num_query, 1, 1, 1)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            query_noise            = torch.randn_like(image_repeat).sign() # Rademacher noise
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []
            for query_idx in range(num_query//num_sub_query):
                sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]
                print("sub_perturbed_image_repeat size:", sub_perturbed_image_repeat.size())
                with torch.no_grad():
                    text_of_sub_perturbed_imgs = _i2t(args, chat, image_tensor=sub_perturbed_image_repeat)
                text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs)
            
            # step 2. estimate grad
            with torch.no_grad():
                perturb_text_token    = clip.tokenize(text_of_perturbed_imgs, truncate=True).to(device)
                perturb_text_features = clip_img_model_vitb32.encode_text(perturb_text_token)
                perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
                perturb_text_features = perturb_text_features.detach()

            print("perturb_text_features size:", perturb_text_features.size())
            print("adv_text_features size:", adv_text_features.size())
            print("tgt_text_features size:", tgt_text_features.size())
            
            coefficient = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)  # size = (num_query * batch_size)
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = coefficient * query_noise / sigma # size = (num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = pseudo_gradient.mean(0) # size = (bs, 3, args.input_res, args.input_res)
            
            # step 3. log metrics
            adv_image_in_current_step = (image + delta)
                
            delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta.data = delta_data
            print(f"img: {i:3d}-step {step_idx} max  delta", torch.max(torch.abs(delta)).item())
            print(f"img: {i:3d}-step {step_idx} mean delta", torch.mean(torch.abs(delta)).item())
            
            # log sim
            with torch.no_grad():
                text_of_adv_image_in_current_step = _i2t(args, chat, image_tensor=adv_image_in_current_step)
                text_token = clip.tokenize(text_of_adv_image_in_current_step, truncate=True).to(device)
                text_features_of_adv_image_in_current_step = clip_img_model_vitb32.encode_text(text_token)
                text_features_of_adv_image_in_current_step = text_features_of_adv_image_in_current_step / text_features_of_adv_image_in_current_step.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step = text_features_of_adv_image_in_current_step.detach()

                adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
                
                # update results
                if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
                    query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
                
                # other clip archs
                # rn50
                tgt_text_features_rn50 = target_text_features_rn50[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn50 = clip_img_model_rn50.encode_text(text_token)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50 / text_features_of_adv_image_in_current_step_rn50.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50.detach()
                adv_txt_tgt_txt_score_in_current_step_rn50 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn50 * tgt_text_features_rn50, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn50 > query_attack_results_rn50[i]:
                    query_attack_results_rn50[i] = adv_txt_tgt_txt_score_in_current_step_rn50
                
                # rn101
                tgt_text_features_rn101 = target_text_features_rn101[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn101 = clip_img_model_rn101.encode_text(text_token)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101 / text_features_of_adv_image_in_current_step_rn101.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101.detach()
                adv_txt_tgt_txt_score_in_current_step_rn101 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn101 * tgt_text_features_rn101, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn101 > query_attack_results_rn101[i]:
                    query_attack_results_rn101[i] = adv_txt_tgt_txt_score_in_current_step_rn101
                
                # vitb16
                tgt_text_features_vitb16 = target_text_features_vitb16[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitb16 = clip_img_model_vitb16.encode_text(text_token)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16 / text_features_of_adv_image_in_current_step_vitb16.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16.detach()
                adv_txt_tgt_txt_score_in_current_step_vitb16 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitb16 * tgt_text_features_vitb16, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitb16 > query_attack_results_vitb16[i]:
                    query_attack_results_vitb16[i] = adv_txt_tgt_txt_score_in_current_step_vitb16
                
                # vitl14
                tgt_text_features_vitl14 = target_text_features_vitl14[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitl14 = clip_img_model_vitl14.encode_text(text_token)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14 / text_features_of_adv_image_in_current_step_vitl14.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14.detach()
                adv_txt_tgt_txt_score_in_current_step_vitl14 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitl14 * tgt_text_features_vitl14, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitl14 > query_attack_results_vitl14[i]:
                    query_attack_results_vitl14[i] = adv_txt_tgt_txt_score_in_current_step_vitl14
                    # ----------------
                
            # # log text
            # with open(os.path.join("../_output_text", args.output + '_pred.txt'), 'a') as f:
            #     print('\n'.join(text_of_adv_image_in_current_step), file=f)
            # f.close()
            
            # # save img
            # os.makedirs(os.path.join('../_output_img', args.output), exist_ok=True)
            # adv_image_to_save = torch.clamp((adv_image_in_current_step) / 255.0, 0.0, 1.0)
            # for path_idx in range(len(path)):
            #     folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            #     folder_to_save = os.path.join('../_output_img', args.output, folder)
            #     if not os.path.exists(folder_to_save):
            #         os.makedirs(folder_to_save, exist_ok=True)
            #     torchvision.utils.save_image(adv_image_to_save[path_idx], os.path.join(folder_to_save, name[:-3] + ".png"))

        if args.wandb:
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
