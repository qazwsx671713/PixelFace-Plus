# PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks
[![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

[PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks](https://dl.acm.org/doi/10.1145/3581783.3612067) 

By Xiaoxiong Du, Jun Peng, Yiyi Zhou, Jinlu Zhang, Siting Chen, Guannan Jiang, Xiaoshuai Sun, Rongrong Ji.

MM '23: Proceedings of the 31st ACM International Conference on Multimedia

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
The paper link：https://dl.acm.org/doi/10.1145/3581783.3612067

The Model link: https://pan.baidu.com/s/1ARSjz6IXCO2-8qf1Tf9p-A?pwd=qwer, the file extraction code:qwer.

Train:
Before training, please dowload the dataset2.json (which has been compressed as a zip file), and place the file in the MMceleba dataset directory.
1. cd /PixelFace+/code
2. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node  2 --master_port 10011  main.py --cfg cfg/mmceleba.yml
