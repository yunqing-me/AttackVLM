import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image

from lavis.models import load_model_and_preprocess


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
    

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        image_processed = vis_processors["eval"](original_tuple[0])
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        
        return image_processed, original_tuple[1], path
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=20, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="tmp", type=str, help='the folder name that restore your outputs')
    
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    args = parser.parse_args()

    alpha = args.alpha
    epsilon = args.epsilon

    # for normalized imgs
    scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
    scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)
    alpha = args.alpha / 255.0 / scaling_tensor
    epsilon = args.epsilon / 255.0 / scaling_tensor

    # select and load model
    print(f"Loading LAVIS models: {args.model_name}, model_type: {args.model_type}...")
    blip_model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    print(f"Done")
    
    # ------------- pre-processing images/text ------------- #
    imagenet_data = ImageFolderWithPaths("/raid/common/imagenet-raw/val/", transform=None)
    target_data   = ImageFolderWithPaths("../../fine-tune/_outputs/sd_coco/", transform=None)
    
    data_loader_imagenet = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    data_loader_target   = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

    # start attack
    for i, ((image_org, _, path), (image_tgt, _, _)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
        if args.batch_size * (i+1) > args.num_samples:
            break
        
        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)
        
        sample_org = {"image": image_org}
        sample_tgt = {"image": image_tgt}
        
        # extract image features
        with torch.no_grad():
            if "blip2" in args.model_name:
                tgt_image_features = blip_model.forward_encoder_image(sample_tgt)             
            else:
                tgt_image_features = blip_model.forward_encoder(sample_tgt)
            tgt_image_features = tgt_image_features[:,0,:]
            tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)
        
        # -------- get adv image -------- #
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
            adv_image          = image_org + delta   # image is normalized to (0.0, 1.0)
            sample_adv         = {"image": adv_image}
            if "blip2" in args.model_name:
                adv_image_features = blip_model.forward_encoder_image(sample_adv)
            else:
                adv_image_features = blip_model.forward_encoder(sample_adv)
                
            adv_image_features = (adv_image_features)[:,0,:]  # size = (bs, 768)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
            
            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))  # cos. sim
            embedding_sim.backward()
            
            grad = delta.grad.detach()
            delta_data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = delta_data
            delta.grad.zero_()
            print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(delta_data)).item():.3f}, mean delta={torch.mean(torch.abs(delta_data)).item():.3f}")

        # save imgs
        adv_image = image_org + delta
        adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
        
        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join('../_output_img', args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + 'png')
