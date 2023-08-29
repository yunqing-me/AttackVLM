# #!/bin/bash


# CUDA_VISIBLE_DEVICES=5 python _train_adv_img_blip.py  \
# --batch_size 50 \
# --num_samples 10000 \
# --steps 150 \
# --output "blip_1_adv" \
# --model_name "blip_caption" \
# --model_type "base_coco" \
# & \
# CUDA_VISIBLE_DEVICES=6 python _train_adv_img_blip.py  \
# --batch_size 25 \
# --num_samples 10000 \
# --steps 150 \
# --output "blip_2_adv" \
# --model_name "blip2_opt" \
# --model_type "pretrain_opt2.7b" \
# & \
# CUDA_VISIBLE_DEVICES=7 python _train_adv_img_blip.py  \
# --batch_size 25 \
# --num_samples 10000 \
# --steps 100 \
# --output "img2llm_adv_2" \
# --model_name "blip_caption" \
# --model_type "large_coco" \
# & \


CUDA_VISIBLE_DEVICES=0 python _train_adv_img_blip.py  \
--batch_size 25 \
--num_samples 1000 \
--steps 50 \
--output "blip_2_adv_pgd50" \
--model_name "blip2_opt" \
--model_type "pretrain_opt2.7b" \
& \
CUDA_VISIBLE_DEVICES=1 python _train_adv_img_blip.py  \
--batch_size 25 \
--num_samples 1000 \
--steps 10 \
--output "blip_2_adv_pgd10" \
--model_name "blip2_opt" \
--model_type "pretrain_opt2.7b" \