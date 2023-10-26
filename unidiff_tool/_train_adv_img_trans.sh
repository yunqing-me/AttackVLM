# #!/bin/bash


# certain hyper-parameters can be modified based on user's preference
python _train_adv_img_trans.py \
    --output 'dir of output white-box transfer images' \
    --epsilon 8 \
    --batch_size 250 \
    --num_samples 1000 \
    --steps 300 \