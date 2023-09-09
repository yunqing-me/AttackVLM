# #!/bin/bash


CUDA_VISIBLE_DEVICES=5 python _train_adv_img_query.py \
--text_path '../_output_text/minigpt4_adv_pred.txt' \
--batch_size 1 \
--num_samples 1000 \
--steps 8 \
--sigma 8 \
--delta 'zero' \
--num_query 30 \
--num_sub_query 10 \
--wandb \
--wandb_project_name minigpt4_adv_query \
--wandb_run_name sigma_8_zero_delta \