

## Prepare the VLMs in LAVIS Lib

There are two steps of adversarial attack for VLMs: (1) transfer-based attacking strategy and (2) query-based attacking strategy for the further improvement.

### Building a suitable LAVIS environment
```
conda create -n lavis python=3.8
conda activate lavis

git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```
or following the steps [HERE](https://github.com/salesforce/LAVIS), and you can refer to the [ModelZoo](https://opensource.salesforce.com/LAVIS//latest/getting_started.html#model-zoo) for the possible model candidates.

## <b> Example: BLIP </b>

Here, we use BLIP for an example. For other models supported in the LAVIS library, please refer to their ```bash``` script (BLIP2, Img2Prompt, etc.) with similar commands as BLIP.
### Transfer-based attacking strategy

```
python _train_adv_img_blip.py  \
    --batch_size 25 \
    --num_samples 1000 \
    --steps 100 \
    --output 'dir of white-box transfer images' \
    --model_name blip_caption \
    --model_type base_coco
```
the crafted adv images x_trans will be stored in `dir of white-box transfer images`. For other models like `BLIP2`, see `bash adv_img_transfer_blip.sh`. Then, we perform image-to-text and store the generated response of x_trans. This can be achieved by:

```
python _lavis_img2txt.py  \
    --batch_size 50 \
    --num_samples 1000 \
    --img_path 'dir of white-box transfer images' \
    --output_path 'name (.txt) of white-box transfer captions' \
    --model_name "blip_caption" \
    --model_type "base_coco"
```

where the generated responses will be stored in `name (.txt) of white-box transfer captions`. For other models like `BLIP2`, see `bash_img2txt_blip.sh`.
We will use them for pseudo-gradient estimation via RGF-estimator. 

### Query-based attacking strategy (via RGF-estimator)

We use `BLIP` as an example script, refer to the following:
```
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
```

Note that this is for fixed perturbation budget (e.g., 8 px) of `MF-ii+MF-tt`. If you plan to assign more budget, please modify the corresponding hyper-parameters accordingly (also see other models in `bash_adv_img_query.sh`).