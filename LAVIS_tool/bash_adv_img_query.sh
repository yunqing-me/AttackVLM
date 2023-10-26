# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

# blip
python _train_adv_img_query.py \
    --data_path 'dir of white-box transfer images' \
    --text_path 'name (.txt) of white-box transfer captions' \
    --output_path 'dir/name of queried captions' \
    --model_name blip_caption \
    --model_type base_coco \
    --batch_size 1 \
    --num_samples 1000 \
    --steps 8 \
    --sigma 8 \
    --delta 'zero' \
    --num_query 100 \
    --num_sub_query 50 \
    --wandb \
    --wandb_project_name lavis \
    --wandb_run_name blip

# blip2
python _train_adv_img_query.py \
    --data_path 'dir of white-box transfer images' \
    --text_path 'name (.txt) of white-box transfer captions' \
    --output_path 'dir/name of queried captions' \
    --model_name blip2_opt \
    --model_type pretrain_opt2.7b \
    --batch_size 1 \
    --num_samples 1000 \
    --steps 8 \
    --sigma 8 \
    --delta 'zero' \
    --num_query 100 \
    --num_sub_query 50 \
    --wandb \
    --wandb_project_name lavis \
    --wandb_run_name blip2