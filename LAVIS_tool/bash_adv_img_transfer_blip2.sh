# #!/bin/bash


python _train_adv_img_blip.py  \
    --batch_size 25 \
    --num_samples 10000 \
    --steps 100 \
    --output "blip2_trans" \
    --model_name "blip2_opt" \
    --model_type "pretrain_opt2.7b" 
