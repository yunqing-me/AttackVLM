# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

python _train_adv_img_query_minigpt4.py  \
    --batch_size 10 \
    --num_samples 10000 \
    --steps 100 \
    --output "minigpt4_adv" \