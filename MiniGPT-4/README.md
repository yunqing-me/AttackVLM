

# Installation


### Building a suitable MiniGPT-4 environment
```
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4
```
or following the steps [HERE](https://github.com/Vision-CAIR/MiniGPT-4), and you can refer to the [LLM weights](https://github.com/Vision-CAIR/MiniGPT-4#:~:text=Prepare%20the%20pretrained%20LLM%20weights) for the possible model candidates.

## <b> MiniGPT-4 </b>

[still in update...]

### Transfer-based attacking strategy

```
bash _train_adv_img_transfer.sh
```
the crafted adv images x_trans will be stored in `name_of_your_output_img_folder`. Then, we perform image-to-text and store the generated response of x_trans. This can be achieved by:

```
bash _minigpt4_img2txt.sh
```
where the generated responses will be stored in `name_of_your_output_txt_file.txt`. We will use them for pseudo-gradient estimation via RGF-estimator.

### Query-based attacking strategy (via RGF-estimator)

```
bash _train_adv_img_query.sh
```