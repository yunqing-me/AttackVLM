# #!/bin/bash


python _train_adv_img_query.py \
    --text_path '../_output_text/blip_trans.txt' \
    --model_name blip_caption \
    --model_type base_coco \
    --batch_size 1 \
    --num_samples 10000 \
    --steps 8 \
    --sigma 8 \
    --delta 'zero' \
    --num_query 100 \
    --num_sub_query 50 \
    --wandb \
    --wandb_project_name blip \
    --wandb_run_name trans+query