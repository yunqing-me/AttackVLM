# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference

# blip
python _train_adv_img_blip.py  \
    --batch_size 25 \
    --num_samples 1000 \
    --steps 100 \
    --output 'dir of white-box transfer images' \
    --model_name blip_caption \
    --model_type base_coco

# blip2
python _train_adv_img_blip.py  \
    --batch_size 25 \
    --num_samples 1000 \
    --steps 100 \
    --output 'dir of white-box transfer images' \
    --model_name "blip2_opt" \
    --model_type "pretrain_opt2.7b" 