# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
python _train_adv_img_query.py \
    --output 'dir/name of queried captions' \
    --data_path 'dir of white-box transfer images' \
    --text_path 'name (.txt) of white-box transfer captions' \
    --batch_size 1 \
    --num_samples 1000 \
    --steps 8 \
    --epsilon 8 \
    --sigma 8 \
    --delta 'zero' \
    --num_query 100 \
    --num_sub_query 25 \
    --wandb \
    --wandb_project_name unidiff-attack \
    --wandb_run_name more-perturb-budget-for-query \