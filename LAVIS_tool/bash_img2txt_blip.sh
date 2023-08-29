# #!/bin/bash


python _lavis_img2txt.py  \
    --batch_size 50 \
    --num_samples 10000 \
    --img_path '../_output_img/blip_trans' \
    --output_path "blip_trans" \
    --model_name "blip_caption" \
    --model_type "base_coco"