# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

# blip
python _lavis_img2txt.py  \
    --batch_size 50 \
    --num_samples 1000 \
    --img_path 'dir of white-box transfer images' \
    --output_path 'name (.txt) of white-box transfer captions' \
    --model_name "blip_caption" \
    --model_type "base_coco"


# blip2
python _lavis_img2txt.py  \
    --batch_size 50 \
    --num_samples 1000 \
    --img_path 'dir of white-box transfer images' \
    --output_path 'name (.txt) of white-box transfer captions' \
    --model_name blip2_opt \
    --model_type pretrain_opt2.7b \