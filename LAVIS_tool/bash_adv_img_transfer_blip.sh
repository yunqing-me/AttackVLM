# #!/bin/bash


python _train_adv_img_blip.py  \
    --batch_size 25 \
    --num_samples 10000 \
    --steps 100 \
    --output "blip_trans" \
    --model_name blip_caption \
    --model_type base_coco
