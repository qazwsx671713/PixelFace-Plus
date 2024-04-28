import argparse
import os
import torch
import torchvision
from torch_fidelity import calculate_metrics
import numpy as np
from ipdb import set_trace
import shutil
from tqdm import tqdm
from tensor_transforms import convert_to_coord_format
from torchvision import utils
@torch.no_grad()
def calculate_fid(model, val_dataset,train_dataset, bs, textEnc, num_batches, latent_size,data_iter,
                    prepare_data,get_text_input,word2id,get_text,clip_word_emb,
                  val_loader,save_dir='fid_imgs', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    save_text_path = save_dir + '/text/'
    os.makedirs(save_text_path, exist_ok=True)
    for i in tqdm(range(num_batches*2)):
        try:
            data = data_iter.next()
        except:
            data_iter = iter(val_loader)
            data = data_iter.next()

        imgs, mask,caps, cap_lens, _, keys = prepare_data(data)
        save_texts = get_text(caps)
        # FOR CLIP
        texts = get_text_input(caps)
        word_emb = clip_word_emb(texts,textEnc).detach()
        states = textEnc.encode_text(texts).float()
        states = states.detach()
        noise=[torch.randn(bs//2, 512, device=device)]
        fake_imgs, _, _ = model(mask,word_emb,noise=noise)
        for j in range(bs//2):
            cnt += 1
            img_name = f"{keys[j]}_{str(cnt).zfill(6)}.png"
            torchvision.utils.save_image(fake_imgs[j, :, :, :],
                                         os.path.join(save_dir, img_name), range=(-1, 1),
                                         normalize=True)
            text_name = f"{keys[j]}_{str(cnt).zfill(6)}.txt"
            file = open(os.path.join(save_text_path, text_name),'w')
            file.write(save_texts[j])
            file.close()
    metrics_dict1 = calculate_metrics(input1=save_dir, input2=train_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    metrics_dict2 = calculate_metrics(input1=save_dir, input2=val_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    if os.path.exists(save_dir) is not None:
        shutil.rmtree(save_dir)
    return metrics_dict1,metrics_dict2