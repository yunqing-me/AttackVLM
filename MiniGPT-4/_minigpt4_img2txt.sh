# #!/bin/bash

python _minigpt4_img2txt.py  \
    --batch_size 10 \
    --num_samples 10000 \
    --img_path 'path to mf-ii images' \
    --output_path 'name (.txt) of transfer captions' \