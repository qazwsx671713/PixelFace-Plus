# PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks
[![Python 3.6](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

[PixelFace+: Towards Controllable Face Generation and Manipulation with Text Descriptions and Segmentation Masks](https://dl.acm.org/doi/10.1145/3581783.3612067) 
By Xiaoxiong Du, Jun Peng, Yiyi Zhou, Jinlu Zhang, Siting Chen, Guannan Jiang, Xiaoshuai Sun, Rongrong Ji.
MM '23: Proceedings of the 31st ACM International Conference on Multimedia


The paper link：https://dl.acm.org/doi/10.1145/3581783.3612067

The Model link: https://pan.baidu.com/s/1ARSjz6IXCO2-8qf1Tf9p-A?pwd=qwer, the file extraction code:qwer.

Train:
Before training, please dowload the dataset2.json (which has been compressed as a zip file), and place the file in the MMceleba dataset directory.
1. cd /PixelFace+/code
2. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node  2 --master_port 10011  main.py --cfg cfg/mmceleba.yml
