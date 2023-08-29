# #!/bin/bash


python _lavis_img2txt.py  \
    --batch_size 50 \
    --num_samples 10000 \
    --img_path '../_output_img/blip2_trans' \
    --output_path "blip2_trans" \
    --model_name blip2_opt \
    --model_type pretrain_opt2.7b \