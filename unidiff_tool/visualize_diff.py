
# import module
from PIL import Image, ImageChops

import os
import argparse
import numpy as np
import torch
import torchvision  
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_res", default=224, type=int)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

os.environ["PYTHONHASHSEED"] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
    

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


imagenet_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(args.input_res),
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Lambda(lambda img: img * 255.),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
    ]
)

# assign images
img1 = imagenet_transform(Image.open("../imagenet_resized/n01440764/ILSVRC2012_val_00000293.png"))
img2 = imagenet_transform(Image.open("../imagenet_resized/n01440764/ILSVRC2012_val_00000293.png"))
  
# finding difference
# diff = ImageChops.difference(img1, img2)  # for PIL images
diff = torch.clamp(((img1 - img2)*10) / 255.0, 0.0, 1.0) 
   
# showing the difference
torchvision.utils.save_image(diff, "./output_adv_training/difference_00000293_x.png")