import clip
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from torchvision import transforms

import math
import numpy as np
from miscc.config import cfg
from distributed import (
	get_rank,
	get_world_size,
)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
	"""Returns cosine similarity between x1 and x2, computed along dim.
	"""
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def d_logistic_loss(netD, real_img, fake_img, 
	c_code=None, real_labels=None, fake_labels=None):

	real_pred, cond_real_logits = netD(real_img, c_code)
	fake_pred, cond_fake_logits = netD(fake_img, c_code)

	_, cond_wrong_logits = netD(
		fake_img, 
		torch.cat([c_code[2:], c_code[0:2]], 0)
	)

	real_loss = F.softplus(-real_pred)
	fake_loss = F.softplus(fake_pred)

	bs = real_img.size(0)

	cond_real_loss = nn.BCELoss()(cond_real_logits, real_labels)
	cond_fake_loss = nn.BCELoss()(cond_fake_logits, fake_labels)
	cond_wrong_loss = nn.BCELoss()(cond_wrong_logits, fake_labels)

	d_loss = ((real_loss.mean() + cond_real_loss) / 2. + 
			 (fake_loss.mean() + cond_fake_loss + cond_wrong_loss) / 3.)

	return d_loss, real_pred, fake_pred

def d_r1_loss(netD, real_img, c_code=None):
	real_pred, _ = netD(real_img)
	grad_real, = autograd.grad(
		outputs=real_pred.sum(), inputs=real_img, create_graph=True
	)
	grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
	
	return grad_penalty, real_pred


def pixel_g_nonsaturating_loss(netD, real_img,fake_imgs, c_code, real_labels):
	fake_pred, cond_logits = netD(fake_imgs, c_code)
	real_loss = F.softplus(-fake_pred).mean()

	cond_real_loss = nn.BCELoss()(cond_logits, real_labels)
	
	g_loss = real_loss + cond_real_loss 

	return g_loss,real_loss,cond_real_loss

def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
	noise = torch.randn_like(fake_img) / math.sqrt(
		fake_img.shape[2] * fake_img.shape[3]
	)

	grad, = autograd.grad(
		outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
	)

	path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

	path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

	path_penalty = (path_lengths - path_mean).pow(2).mean()

	return path_penalty, path_mean.detach(), path_lengths


class CLIPLoss(torch.nn.Module):
	def __init__(self, model):
		super(CLIPLoss, self).__init__()
		# RN50 or ViT-B/32
		self.model = model
		self.preprocess = transforms.Resize([224, 224])

	def forward(self, image, text, labels):
		image = self.preprocess(image)
		logits_per_image, logits_per_text = self.model(image, text)
		loss = nn.CrossEntropyLoss()(logits_per_image, labels)
		return loss
