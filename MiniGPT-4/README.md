

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


### Transfer-based attacking strategy

```
bash _train_adv_img_transfer_minigpt4.sh
```
the crafted adv images x_trans will be stored in `../_output_img/name_of_your_output_img_folder`. Then, we perform image-to-text and store the generated response of x_trans. This can be achieved by:

```
bash _minigpt4_img2txt.sh
```
where the generated responses will be stored in `./output_minigpt4/name_of_your_output_txt_file.txt`. We will use them for pseudo-gradient estimation via RGF-estimator.

### Query-based attacking strategy (via RGF-estimator)

```
bash _train_adv_img_query_minigpt4.sh
```



# Bibtex
If you find this project useful in your research, please consider citing our paper:

```
@article{zhao2023evaluate,
  title={On Evaluating Adversarial Robustness of Large Vision-Language Models},
  author={Zhao, Yunqing and Pang, Tianyu and Du, Chao and Yang, Xiao and Li, Chongxuan and Cheung, Ngai-Man and Lin, Min},
  journal={arXiv preprint arXiv:2305.16934},
  year={2023}
}
```

Meanwhile, a relevant research that aims to [Embedding a Watermark to (multi-modal) Diffusion Models](https://github.com/yunqing-me/WatermarkDM):
```
@article{zhao2023recipe,
  title={A Recipe for Watermarking Diffusion Models},
  author={Zhao, Yunqing and Pang, Tianyu and Du, Chao and Yang, Xiao and Cheung, Ngai-Man and Lin, Min},
  journal={arXiv preprint arXiv:2303.10137},
  year={2023}
}
```