# PixelFace+
The official implementation version of PixelFace+ in ACM MultiMedia 2023.
The paper link：https://dl.acm.org/doi/10.1145/3581783.3612067
The Model link: https://pan.baidu.com/s/1ARSjz6IXCO2-8qf1Tf9p-A?pwd=qwer, the file extraction code:qwer.

Train:
1. cd /PixelFace+/code
2. CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node  2 --master_port 10011  main.py --cfg cfg/mmceleba.yml
