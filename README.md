# PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks
[![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

[PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks](https://dl.acm.org/doi/10.1145/3581783.3612067) 

By Xiaoxiong Du, Jun Peng, Yiyi Zhou, Jinlu Zhang, Siting Chen, Guannan Jiang, Xiaoshuai Sun, Rongrong Ji.

MM '23: Proceedings of the 31st ACM International Conference on Multimedia

## DEMO VIDEO
[![Demo Video](https://img.youtube.com/vi/tIKXBXaBbTo/0.jpg)](https://www.youtube.com/watch?v=tIKXBXaBbTo)

## Introduction
This repository is pytorch implementation of PixelFace+. PixelFace+ utilizes both mask and text features for highly controllable face generation and manipulation. We propose the GCMF module to achieve better decoupling. Additionally, to enhance the alignment between generated images and text, we introduce a regularization loss function based on CLIP. The framework diagram of PixelFace+ is shown below:![The Framework of PixelFace+](https://github.com/qazwsx671713/PixelFace-Plus/blob/main/framwork.png)

## Citation
```
@inproceedings{10.1145/3581783.3612067,
author = {Du, Xiaoxiong and Peng, Jun and Zhou, Yiyi and Zhang, Jinlu and Chen, Siting and Jiang, Guannan and Sun, Xiaoshuai and Ji, Rongrong},
title = {PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3612067},
doi = {10.1145/3581783.3612067},
pages = {4666–4677},
numpages = {12},
keywords = {controllable face generation, face editing},
series = {MM '23}
}
```

## Prerequisites
```
python 3.6
pytorch 1.10.0
pytorch-fid 0.2.1
torchvision 0.11.1
```

## Data preparation
Multi-Modal-CelebA-HQ Dataset [[Link](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)]

Before training, please dowload the dataset2.json (which has been compressed as a zip file), and place the file in the MMceleba dataset directory.
## Training
1. Preparing your settings. To train a model, you should modify code/cfg/mmceleba.yml to adjust the settings you want. The default configuration is to train on MMceleba with input and output image resolution set to 256*256, and BatchSize set to 4. **Increasing the BatchSize may result in a decrease in semantic alignment after training, as a larger BatchSize reduces the constraint of the CLIP regularization loss.**

2. Training the model. run  train.py under the main folder to start training:
```
cd /PixelFace+/code
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node  2 --master_port 10011  main.py --cfg cfg/mmceleba.yml
```
3. Testing the model. After training for more than 70 epochs, the model automatically evaluates its performance every ten epochs. If you need to modify the evaluation frequency, you can do so at line 675 in `\code\trainer.py`.

## Pretrain Model
1. Dowload the pretrain model.
The Model link: https://pan.baidu.com/s/1ARSjz6IXCO2-8qf1Tf9p-A?pwd=qwer, the file extraction code:qwer.

2. Modify the cfg file`\code\cfg\mmceleba.yml` to use the pretrain model:
```
TRAIN:
  FLAG: True

  ##### Modify This Line #####
  NET_G: '/PATH/TO/PRETRAIN/MODEL'

  B_NET_D: True
  BATCH_SIZE: 4  
  MAX_EPOCH: 100
  SNAPSHOT_INTERVAL: 1  
  DISCRIMINATOR_LR: 0.004
  GENERATOR_LR: 0.002
```
## Acknowledgement
Thanks for a lot of codes from [PixelFolder](https://github.com/BlingHe/PixelFolder) and [PixelFace](https://github.com/pengjunn/PixelFace).
