# #!/bin/bash


CUDA_VISIBLE_DEVICES=2 python _train_adv_img_minigpt4.py  \
--batch_size 10 \
--num_samples 10000 \
--steps 100 \
--output "minigpt4_adv" \