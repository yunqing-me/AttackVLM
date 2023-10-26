import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image


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
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=300, type=int)
    parser.add_argument("--output", default="temp", type=str, help='the folder name of output')
    
    parser.add_argument("--cle_data_path", default=None, type=str, help='path of the clean images')
    parser.add_argument("--tgt_data_path", default=None, type=str, help='path of the target images')
    args = parser.parse_args()
    
    # load clip_model params
    alpha = args.alpha
    epsilon = args.epsilon
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)
     
    # ------------- pre-processing images/text ------------- #
    
    # preprocess images
    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(args.input_res),
            torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
            torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )
    clean_data    = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    target_data   = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
    
    data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=False)
    data_loader_target   = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=12, drop_last=False)

    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(clip_model.visual.input_resolution, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
    
    # CLIP imgs mean and std.
    inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])

    # start attack
    for i, ((image_org, _, path), (image_tgt, _, _)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
        if args.batch_size * (i+1) > args.num_samples:
            break
        
        # (bs, c, h, w)
        image_org = image_org.to(device)
        image_tgt = image_tgt.to(device)
        
        # get tgt featutres
        with torch.no_grad():
            tgt_image_features = clip_model.encode_image(clip_preprocess(image_tgt))
            tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)

        # -------- get adv image -------- #
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
            adv_image = image_org + delta
            adv_image = clip_preprocess(adv_image)
            adv_image_features = clip_model.encode_image(adv_image)
            adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

            embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))  # computed from normalized features (therefore it is cos sim.)
            embedding_sim.backward()
            
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
            delta.data = d
            delta.grad.zero_()
            print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(d)).item():.3f}, mean delta={torch.mean(torch.abs(d)).item():.3f}")

        # save imgs
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join(args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            if 'JPEG' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + 'png')
            elif 'png' in name:
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name))

