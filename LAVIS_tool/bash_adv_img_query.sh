# #!/bin/bash



# CUDA_VISIBLE_DEVICES=4 python _train_adv_img_query.py \
# --text_path '../_output_text/blip_1_adv_pred.txt' \
# --model_name blip_caption \
# --model_type base_coco \
# --batch_size 1 \
# --num_samples 1000 \
# --steps 8 \
# --sigma 8 \
# --delta 'zero' \
# --num_query 50 \
# --num_sub_query 50 \
# --wandb \
# --wandb_project_name blip_1_adv_query \
# --wandb_run_name sigma_8_zero_delta \
# & \
# CUDA_VISIBLE_DEVICES=5 python _train_adv_img_query.py \
# --text_path '../_output_text/blip_2_adv_pred.txt' \
# --model_name blip2_opt \
# --model_type pretrain_opt2.7b \
# --batch_size 1 \
# --num_samples 1000 \
# --steps 8 \
# --sigma 8 \
# --delta 'zero' \
# --num_query 50 \
# --num_sub_query 50 \
# --wandb \
# --wandb_project_name blip_2_adv_query \
# --wandb_run_name sigma_8_zero_delta \
# & \
# CUDA_VISIBLE_DEVICES=6 python _train_adv_img_query.py \
# --text_path '../_output_text/img2llm_adv_pred.txt' \
# --model_name img2prompt_vqa \
# --model_type base \
# --batch_size 1 \
# --num_samples 1000 \
# --steps 8 \
# --sigma 8 \
# --delta 'zero' \
# --num_query 30 \
# --num_sub_query 10 \
# --wandb \
# --wandb_project_name img2prompt_adv_query \
# --wandb_run_name sigma_8_zero_delta \


# rebuttal 
# CUDA_VISIBLE_DEVICES=2 python _train_adv_img_query.py \
# --text_path '../_output_text/blip_2_adv_pgd10_pred.txt' \
# --model_name blip2_opt \
# --model_type pretrain_opt2.7b \
# --batch_size 1 \
# --num_samples 1000 \
# --steps 8 \
# --sigma 8 \
# --delta 'zero' \
# --num_query 50 \
# --num_sub_query 50 \
# --wandb \
# --wandb_project_name rebuttal_blip_2_adv_query \
# --wandb_run_name pgd10 \
# & \
# CUDA_VISIBLE_DEVICES=3 python _train_adv_img_query.py \
# --text_path '../_output_text/blip_2_adv_pgd50_pred.txt' \
# --model_name blip2_opt \
# --model_type pretrain_opt2.7b \
# --batch_size 1 \
# --num_samples 1000 \
# --steps 8 \
# --sigma 8 \
# --delta 'zero' \
# --num_query 50 \
# --num_sub_query 50 \
# --wandb \
# --wandb_project_name rebuttal_blip_2_adv_query \
# --wandb_run_name pgd50 \


CUDA_VISIBLE_DEVICES=2 python _train_adv_img_query.py \
--text_path '../_output_text/blip_2_imagenet_pred.txt' \
--model_name blip2_opt \
--model_type pretrain_opt2.7b \
--batch_size 1 \
--num_samples 1000 \
--steps 8 \
--sigma 8 \
--delta 'zero' \
--num_query 50 \
--num_sub_query 50 \
--wandb \
--wandb_project_name blip_2_query_only \
--wandb_run_name mf-tt \