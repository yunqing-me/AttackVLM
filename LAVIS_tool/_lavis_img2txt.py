import argparse
import os
import random

import torch
import torchvision
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

# from lavis.common.gradcam import getAttMap
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


class ImageFolderForLavis(torchvision.datasets.ImageFolder):
    def __init__(self, root, processor, transform = None, target_transform = None, loader = ..., is_valid_file = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.processor = processor
        
    def __getitem__(self, index: int):
        
        # original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        image   = self.processor["eval"](Image.open(path).convert('RGB'))
        
        return (image, path)
    

def main():
    seedEverything()
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    
    parser.add_argument("--img_path", default='/raid/common/imagenet-raw/val/', type=str)
    parser.add_argument("--output_path", default="lavis_tmp", type=str)
    
    args = parser.parse_args()

    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    print(f"Loading LAVIS models: {args.model_name}, model_type: {args.model_type}...")
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    print("Done.")
    
    dataset    = ImageFolderForLavis(args.img_path, processor=vis_processors, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=24)

    for i, (image, path) in enumerate(dataloader):
        start = time.perf_counter()
        print(f"LAVIS img2txt: {i}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples//args.batch_size:
            break
        
        image = image.to(device)
        # generate caption
        if args.model_name == "img2prompt_vqa":
            question = "what is the content of this image?"
            question = txt_processors["eval"](question)
            samples  = {"image": image, "text_input": [question]*args.batch_size}
            
            # obtain gradcam and update dict
            with torch.no_grad():
                samples  = model.forward_itm(samples=samples)
            samples  = model.forward_cap(samples=samples, num_captions=1, num_patches=20)
            caption  = samples['captions']
            for cap_idx, cap in enumerate(caption):
                if cap_idx == 0:
                    caption_merged = cap
                else:
                    caption_merged = caption_merged + cap
        else:
            with torch.no_grad():
                samples  = {"image": image}
                caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
                caption_merged = caption
                
        # write caption
        with open(os.path.join("../_output_text", args.output_path + '_pred.txt'), 'a') as f:
            print('\n'.join(caption_merged), file=f)
        f.close()
        
        end = time.perf_counter()
        print(f"query time for {args.batch_size} samples:", (end - start))
        
    print("Caption saved.")
    
        
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()