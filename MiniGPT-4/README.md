

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

Please adapt our code of crafting adversarial samples shown in the main page to the MiniGPT-4 training / inference code as the principal ideas are the similar: Firstly we apply transfer-based attack through the visual encoder (it is often a pretraind Clip encoder), then we conduct the black-box attack on the LLM part of that VLM.

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