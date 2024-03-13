# ADD

**Generating Progressive Images from Pathological Transitions via Diffusion Model**<br/>
[Zeyu Liu](https://github.com/Rowerliu), Tianyi Zhang, Yufang He, Yu Zhao, Yunlu Feng, Guanglei Zhang<br/>
[Arxiv](https://arxiv.org/abs/2311.12316), [GitHub](https://github.com/Rowerliu/ADD)

<img src="assets/ADD.png" height="300" />

## 🗃️ Overview
Deep learning is widely applied in computer-aided pathological diagnosis, which alleviates the pathologist workload 
and provide timely clinical analysis. However, most models generally require large-scale annotated data for training, 
which faces challenges due to the sampling and annotation scarcity in pathological images. The rapid developing 
generative models shows potential to generate more training samples from recent studies. However, they also struggle in
generalization diversity with limited training data, incapable of generating effective samples. Inspired by the 
pathological transitions be-tween different stages, we propose an adaptive depth-controlled diffusion (ADD) network
to generate pathological progressive images for effective da-ta augmentation. This novel approach roots in domain migration,
where a hybrid attention strategy guides the bidirectional diffusion, blending local and global attention priorities. 
With feature measuring, the adaptive depth-controlled strategy ensures the migration and maintains locational similarity 
in simulating the pathological feature transition. Based on tiny training set (samples ≤ 500), the ADD yields cross-domain 
progressive images with corresponding soft-labels. Experiments on two datasets suggest significant improvements in 
generation diversity, and the effectiveness with generated progressive samples are highlighted in downstream classifications.

## 🗃️ Usage

### Generating a sequence of intermediate images between source domain and target domain
1. Train a diffusion model on your data based on the [guided-diffusion](https://github.com/openai/guided-diffusion)<br/>
2. Assign the path of trained models, and then generate intermediate images
(The total diffusion process includes 1000 steps, and we can get 10 intermediate images)<br/>
```bash
python scripts/frequency_generating_m_samples.py --diffusion_steps=1000 --amount=10`
```

## 🗃️ Acknowledgements
This implementation is based on / inspired by:<br/>
[openai/guided-diffusion](https://github.com/openai/guided-diffusion)<br/>
[openai/improved-diffusion](https://github.com/openai/improved-diffusion)<br/>
[suxuann/ddib](https://github.com/suxuann/ddib)

## 🗃️ Materials
The comparison methods are listed here:


| Model  | Based method | Paper                                                                                                          | Code                                                         |
|:-------|:-------------|:---------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------|
| ProGAN | GAN          | [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) | [Github](https://github.com/sidward14/gan-lab)                 |
| IDDPM  | Diffusion    | [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)                          | [Github](https://github.com/openai/improved-diffusion)       |
| LoFGAN | GAN          | [LoFGAN: Fusing Local Representations for Few-shot Image Generation](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_LoFGAN_Fusing_Local_Representations_for_Few-Shot_Image_Generation_ICCV_2021_paper.pdf)         | [Github](https://github.com/edward3862/LoFGAN-pytorch)       |
| MixDL  | GAN          | [Towards Expert-Level Medical Question Answering with Large Language Models](https://arxiv.org/abs/2111.11672) | [Github](https://github.com/reyllama/mixdl)                  |
