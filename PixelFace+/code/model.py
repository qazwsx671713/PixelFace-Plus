import math
import random
import functools
import operator
import numpy as np
from ipdb import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from spectral import SpectralNorm
from miscc.config import cfg
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from distributed import get_rank
from email.errors import NonPrintableDefect
import math
from pickle import NONE
import random
import re
from ipdb import set_trace
from numpy import concatenate

import torch
from torch import nn
import torch.nn.functional as F
import sys

from pixel_models.blocks import ConstantInput, StyledConv, ToRGB, PixelNorm, EqualLinear, Unfold, LFF, Con_PosFold,Upsample
import tensor_transforms as tt
from pixel_models.GateCrossModelFusion import GateCrossModelFusion
def float_dtype():
    return torch.float32
	
class GLU(nn.Module):
	def __init__(self):
		super(GLU, self).__init__()

	def forward(self, x):  # (N,c_dim*4)
		nc = x.size(1)  # c_dim*4
		assert nc % 2 == 0, 'channels dont divide 2!'
		nc = int(nc/2)  # c_dim*2
		return x[:, :nc] * torch.sigmoid(x[:, nc:])  # c_dim*2

def make_kernel(k):
	k = torch.tensor(k, dtype=torch.float32)

	if k.ndim == 1:
		k = k[None, :] * k[:, None]

	k /= k.sum()

	return k

class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor=1):
		super().__init__()

		kernel = make_kernel(kernel)

		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)

		self.register_buffer('kernel', kernel)

		self.pad = pad

	def forward(self, input):
		out = upfirdn2d(input, self.kernel, pad=self.pad)

		return out


class EqualConv2d(nn.Module):
	def __init__(
		self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
	):
		super().__init__()

		self.weight = nn.Parameter(
			torch.randn(out_channel, in_channel, kernel_size, kernel_size)
		)
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))

		else:
			self.bias = None

	def forward(self, input):
		out = F.conv2d(
			input,
			self.weight * self.scale,
			bias=self.bias,
			stride=self.stride,
			padding=self.padding,
		)

		return out

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)

class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		out = F.leaky_relu(input, negative_slope=self.negative_slope)

		return out * math.sqrt(2)


class ConvLayer(nn.Sequential):
	def __init__(
		self,
		in_channel,
		out_channel,
		kernel_size,
		downsample=False,
		blur_kernel=[1, 3, 3, 1],
		bias=True,
		activate=True,
	):
		layers = []

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2
			self.padding = 0

		else:
			stride = 1
			self.padding = kernel_size // 2

		layers.append(
			EqualConv2d(
				in_channel,
				out_channel,
				kernel_size,
				padding=self.padding,
				stride=stride,
				bias=bias and not activate,
			)
		)

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))

			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(
			in_channel, out_channel, 1, downsample=True, activate=False, bias=False
		)

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class Pixel_Discriminator(nn.Module):
	def __init__(
		self,
		size,
		channel_multiplier=2, 
		blur_kernel=[1, 3, 3, 1],
		input_size=3, 
		n_first_layers=0,
		**kwargs
	):
		super().__init__()
		
		self.input_size = input_size

		channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

		#convs = [ConvLayer(3, channels[size], 1)]
		convs = [ConvLayer(input_size, channels[size], 1)]
		convs.extend([ConvLayer(channels[size], channels[size], 3) for _ in range(n_first_layers)])

		log_size = int(math.log(size, 2))

		in_channel = channels[size]

		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]

			convs.append(ResBlock(in_channel, out_channel, blur_kernel))

			in_channel = out_channel

		self.convs = nn.Sequential(*convs)

		self.stddev_group = 4
		self.stddev_feat = 1

		self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
		self.final_linear = nn.Sequential(
			EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
			EqualLinear(channels[4], 1),
		)
		nef = cfg.TEXT.EMBEDDING_DIM
		
		self.COND_DNET = D_GET_LOGITS(channels[4], nef, bcondition=True)

	def forward(self, image, c_code=None):
		out = self.convs(image)   

		batch, channel, height, width = out.shape  # N,512,4,4
		group = min(batch, self.stddev_group)
		stddev = out.view(
			group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
		)
		stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
		stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
		stddev = stddev.repeat(group, 1, height, width)
		out = torch.cat([out, stddev], 1)

		out = self.final_conv(out)
		if c_code is not None:
			cond_logits = self.COND_DNET(out, c_code)
		else:
			cond_logits = None
			
		out = out.view(batch, -1)
		out = self.final_linear(out)
		
		return out, cond_logits
# ############## D networks ##########################
def conv3x3(in_planes, out_planes, bias=False):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


def Block3x3_leakRelu(in_planes, out_planes):
	block = nn.Sequential(
		SpectralNorm(conv3x3(in_planes, out_planes, bias=True)),
		nn.LeakyReLU(0.2, inplace=True)
	)
	return block

class D_GET_LOGITS(nn.Module):
	def __init__(self, ndf, nef, bcondition=False):
		super(D_GET_LOGITS, self).__init__()
		self.df_dim = ndf
		self.ef_dim = nef
		self.bcondition = bcondition
		if self.bcondition:
			self.jointConv = Block3x3_leakRelu(ndf + nef, ndf)

		self.outlogits = nn.Sequential(
			nn.Conv2d(ndf, 1, kernel_size=4, stride=4),
			nn.Sigmoid())

	def forward(self, h_code, c_code=None):
		if self.bcondition and c_code is not None:
			# conditioning output
			c_code = c_code.view(-1, self.ef_dim, 1, 1)
			c_code = c_code.repeat(1, 1, 4, 4)
			# state size (ngf+egf) x 4 x 4
			h_c_code = torch.cat((h_code, c_code), 1)
			# state size ngf x in_size x in_size
			h_c_code = self.jointConv(h_c_code)
		else:
			h_c_code = h_code

		output = self.outlogits(h_c_code)
		return output.view(-1)

class D_GET_LOGITS_MASK_COND(nn.Module):
	def __init__(self, ndf, nef, bcondition=False):
		super(D_GET_LOGITS_MASK_COND, self).__init__()
		self.df_dim = ndf
		self.ef_dim = nef
		self.bcondition = bcondition
		if self.bcondition:
			self.jointConv = Block3x3_leakRelu(ndf + nef, ndf)

		self.outlogits = nn.Sequential(
			nn.Conv2d(ndf, 1, kernel_size=4, stride=4),
			nn.Sigmoid())

	def forward(self, h_code, c_code=None):
		if self.bcondition and c_code is not None:
			# conditioning output
			# state size (ngf+egf) x 4 x 4
			h_c_code = torch.cat((h_code, c_code), 1)
			# state size ngf x in_size x in_size
			h_c_code = self.jointConv(h_c_code)
		else:
			h_c_code = h_code

		output = self.outlogits(h_c_code)
		return output.view(-1)


class PixelFacePlus(nn.Module):
    "merge: concatenate"
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        text_dim,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        activation=None,
        hidden_res=16,
        mask_dim = 3,
        **kwargs
    ):
        super().__init__()

        self.size = size

        # ---------------- mappling block -----------------
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            if i == 0:
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )  
            else:
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
        self.style = nn.Sequential(*layers)
        self.mask_cov = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.text_dim=text_dim
        linears = [PixelNorm()]
        linears.append(
                EqualLinear(
                    text_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        linears.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.linears=nn.Sequential(*linears)
        self.fc1 = EqualLinear(
           text_dim , text_dim//2, lr_mul=lr_mlp, activation='fused_lrelu'
        )

        self.style = nn.Sequential(*layers)
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # ------------------ Synthesis block -----------------
        self.log_size = int(math.log(size, 2))  # 8
        self.num_fold_per_stage = 2
        self.num_stage = self.log_size // self.num_fold_per_stage - 1  # 3

        self.posfolders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.lffs = nn.ModuleList()

        in_res = 4
        in_shape = (self.channels[in_res], in_res, in_res)
        
        self.input = ConstantInput(self.channels[4*2**self.num_fold_per_stage], size=4*2**self.num_fold_per_stage)

        for i in range(self.num_stage):
            out_res = in_res * (2**self.num_fold_per_stage)
            out_shape = (self.channels[out_res], out_res, out_res)
            self.lffs.append(LFF(self.channels[out_res]))
            self.posfolders.append(
                Con_PosFold(in_shape=in_shape, out_shape=out_shape,i=i, use_const=True if i==0 else False,mask_dim = mask_dim+128)
            )
            in_channel = self.channels[in_res]
            for fold in range(self.num_fold_per_stage):
                out_channel = self.channels[in_res*(2**(fold+1))]
                self.convs.append(
                    StyledConv(
                    in_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.convs.append(
                    StyledConv(
                    out_channel//4, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.to_rgbs.append(ToRGB(out_channel, style_dim))
                in_channel = out_channel
            in_res = out_res
            in_shape = out_shape
        self.lr_mlp=lr_mlp
        self.unfolder = Unfold()
        self.n_latent = (self.num_fold_per_stage * self.num_stage) * 2 + 1
        kwargs1 = {
                "from_len":  16*16, "to_len":  self.n_latent,   # The from/to tensor lengths
                "from_dim":  128,  "to_dim":  512,             # The from/to tensor dimensions
                "pos_dim": 0                       
            }
        kwargs2 = {
                "from_len":  64*64, "to_len":  self.n_latent,   # The from/to tensor lengths
                "from_dim":  128,  "to_dim":  512,             # The from/to tensor dimensions
                "pos_dim": 0                       
            }
        kwargs3 = {
                "from_len":  256*256, "to_len":  self.n_latent,   # The from/to tensor lengths
                "from_dim":  128,  "to_dim":  512,             # The from/to tensor dimensions
                "pos_dim": 0                   
            }
        self.gcmf = []
        self.gcmf1 = GateCrossModelFusion(dim = 128, **kwargs1)
        self.gcmf2 = GateCrossModelFusion(dim = 128, **kwargs2)
        self.gcmf3 = GateCrossModelFusion(dim = 128, **kwargs3)
        self.gcmf.append(self.gcmf1)
        self.gcmf.append(self.gcmf2)
        self.gcmf.append(self.gcmf3)
        self.blur_kernel=[1, 3, 3, 1]
        self.upsample = Upsample(self.blur_kernel, factor=4)
        print("# PixelFace+")

    def forward(
        self,
        mask,
        c_code, 
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        sentence=None,
        randomize_noise=True,
        return_all_images=False
    ):
    # -------------- mapping blocks -----------
        if len(c_code.shape)!=3:
            raise Exception("Invalid c_code shape!")
        self.gcmf1.to_len=c_code.shape[-1]
        self.gcmf2.to_len=c_code.shape[-1]
        self.gcmf3.to_len=c_code.shape[-1]
        c_code = c_code.permute(0,2,1).contiguous()
        c_shape = c_code.shape
        c_code = c_code.reshape(c_shape[0]*c_shape[1],-1)
        c_code = c_code.type(torch.float)
        c_code = self.linears(c_code)
        c_code = c_code.reshape(c_shape[0],c_shape[1],-1)
        if noise is None:
            raise Exception("noise can not be None")
        cat_noise = []

        if not input_is_latent:
            styles = [self.style(s) for s in noise]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        elif len(styles) > 2:
            latent = torch.stack(styles, 1)

        else:
            
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
    # -------------- synthesis blocks -----------
        images = []
        out = self.input(latent)
        skip = None
        attnmap_img = None
        attn_prob = None
        mask_cov_res = self.mask_cov(mask[2])
        mask_list = []
        mask_list.append(F.interpolate(mask_cov_res,size=[16,16],mode='nearest'))
        mask_list.append(F.interpolate(mask_cov_res,size=[64,64],mode='nearest'))
        mask_list.append(mask_cov_res)
        for i in range(self.num_stage):
            b, _, h, w = out.shape
            if i > 0:
                h, w = h*2**self.num_fold_per_stage, w*2**self.num_fold_per_stage
            
            coord = tt.convert_to_coord_format(b, h, w, device=out.device)

            # Fourior embedding
            emb = self.lffs[i](coord)

            if i == 0:
                att_vars=None
                temp = mask_list[0]
                shape = temp.shape
                temp = temp.reshape(shape[0],shape[1],shape[2]*shape[3]).permute(0, 2, 1)
                temp, att_map, att_vars = self.gcmf[i](
                    from_tensor = temp, #latent
                    to_tensor = c_code, #temp
                    from_pos = None,
                    to_pos =  None,
                    att_vars = att_vars,
                    att_mask = None,
                    hw_shape = shape[-2:] #[(latent.shape)[-2]]
                )
                temp = temp.permute(0, 2, 1).reshape(shape)
                temp = torch.cat((temp,emb), 1)
                temp = torch.cat((temp,out), 1)
                temp = torch.cat((temp, mask_list[i]), 1)
                out = self.posfolders[i](emb, out, temp, is_first=True)
            else:
                temp = mask_list[i]
                shape = temp.shape
                temp = temp.reshape(shape[0],shape[1],shape[2]*shape[3]).permute(0, 2, 1)
                temp, att_map, att_vars = self.gcmf[i](
                    from_tensor = temp,#latent
                    to_tensor = c_code, #temp
                    from_pos = None,
                    to_pos =  None,
                    att_vars = att_vars,
                    att_mask = None,
                    hw_shape = shape[-2:] #[(latent.shape)[-2]]
                )
                temp = temp.permute(0, 2, 1).reshape(shape)
                temp = torch.cat((temp,emb), 1)
                temp=torch.cat((temp, mask_list[i]), 1)
                out = self.posfolders[i](temp, out, is_first=False)
            for fold in range(self.num_fold_per_stage):
                out = self.convs[i*self.num_fold_per_stage*2 + fold*2](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2]
                    )
                out = self.unfolder(out)
                out = self.convs[i*self.num_fold_per_stage*2 + fold*2 + 1](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 1]
                )
                skip = self.to_rgbs[i*self.num_fold_per_stage + fold](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 2], skip
                    )
            images.append(skip)
            
        image = skip
        if return_latents:
            return image, latent ,None
        elif return_all_images:
            return image, images,F.interpolate(mask_cov_res,size=[4,4],mode='nearest')
        else:
            return image, None,None