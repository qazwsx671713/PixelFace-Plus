from __future__ import print_function
import os
import time
import numpy as np
import sys
import shutil
from PIL import Image
from six.moves import range
from tqdm import tqdm
import math
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torchvision
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
import PIL.Image as Image
import sys
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.losses import d_logistic_loss, d_r1_loss
from miscc.losses import g_path_regularize,pixel_g_nonsaturating_loss
from miscc.losses import CLIPLoss 
from datasets import TextDataset, prepare_data,EvalDataset
from model_base import RNN_ENCODER, CNN_ENCODER
sys.path.append("./pixel_models")
from model import Pixel_Discriminator as D_NET
from model import PixelFacePlus as G_STYLE
import pixel_models.tensor_transforms as tt
from calculate_fid import calculate_fid

from distributed import (
	get_rank,
	reduce_loss_dict,
	reduce_sum,
	get_world_size,
	cleanup_distributed, 
)



# ################# Text to image task############################ #
class condGANTrainer(object):
	def __init__(self, output_dir, args):
		
		if cfg.TRAIN.FLAG:
			self.out_dir = output_dir
			self.model_dir = os.path.join(output_dir, 'Model')
			self.image_dir = os.path.join(output_dir, 'Image')
			self.log_dir = os.path.join(output_dir, 'Code_backup')
			mkdir_p(self.model_dir)
			mkdir_p(self.image_dir)
			mkdir_p(self.log_dir)
			
			self.writer = SummaryWriter(output_dir)

			shutil.copy(args.cfg_file, self.log_dir)
			bkfiles = ['datasets', 'main', 'trainer', 'model', 'model_base', 'miscc/losses']
			for _file in bkfiles:
				shutil.copy(f'./{_file}.py', self.log_dir)

			split_dir, bshuffle = 'train', True
		else:
			split_dir, bshuffle = 'test', False
		
		self.args = args
		self.batch_size = cfg.TRAIN.BATCH_SIZE
		self.max_epoch = cfg.TRAIN.MAX_EPOCH   # 800
		self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
		self.img_size = cfg.TREE.BASE_SIZE

		self.data_set = TextDataset(
			cfg.DATA_DIR, 
			split_dir,
			base_size=self.img_size,
		)
		self.data_sampler = self.data_sampler(
			self.data_set, 
			shuffle=bshuffle, 
			distributed=args.distributed
		)
		self.data_loader = data.DataLoader(
			self.data_set, 
			batch_size=self.batch_size,
			sampler=self.data_sampler,
			drop_last=True, 
		)

		self.n_words = self.data_set.n_words
		self.ixtoword = self.data_set.ixtoword  # dict for idx to word
		self.word2id = self.data_set.wordtoix
		self.pretrained_emb = self.data_set.pretrained_emb
		self.num_batches = len(self.data_loader)

		self.path_batch_shrink = cfg.TRAIN.PATH_BATCH_SHRINK
		self.path_batch = max(1, self.batch_size // self.path_batch_shrink)
		if cfg.TRAIN.FLAG:
			self.path_loader = data.DataLoader(
				self.data_set, 
				batch_size=self.path_batch,
				sampler=self.data_sampler,
				drop_last=True, 
			)

			self.val_set = TextDataset(
				cfg.DATA_DIR, 
				'test',
				base_size=self.img_size,
			)
			self.val_loader = data.DataLoader(
				self.val_set, 
				batch_size=self.batch_size,
				drop_last=True, 
				shuffle=False, 
				num_workers=int(cfg.WORKERS)
			)
			self.gen_set = TextDataset(
				cfg.DATA_DIR, 
				'test',
				base_size=self.img_size,
			)
			self.gen_loader = data.DataLoader(
				self.gen_set, 
				batch_size=self.batch_size//2,
				drop_last=True, 
				shuffle=False, 
				num_workers=int(cfg.WORKERS)
			)
			self.eval_val_set=EvalDataset(
				cfg.DATA_DIR, 
				'test',
				base_size=self.img_size,
			)
			self.eval_val_loader = data.DataLoader(
				self.eval_val_set, 
				batch_size=self.batch_size,
				drop_last=True, 
				shuffle=False, 
			)
			self.eval_data_set = EvalDataset(
			cfg.DATA_DIR, 
			'train',
			base_size=self.img_size,
			)

		

	def data_sampler(self, dataset, shuffle, distributed):
		if distributed:
			return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

		elif shuffle:
			return data.RandomSampler(dataset)

		else:
			return data.SequentialSampler(dataset)


	def sample_data(self, loader):
		while True:
			for batch in loader:
				yield batch


	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag


	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


	def build_models(self):
		def count_parameters(model):
			total_param = 0
			for name, param in model.named_parameters():
				if param.requires_grad:
					num_param = np.prod(param.size())
					total_param += num_param
			return total_param

		device = self.args.device
		
		# ###################encoders################## #
		# init and load text encoder
		if get_rank() == 0:
			print('Load text encoder from:', "CLIP")
			print('Load image encoder from:', "CLIP")
		self.clip_model, _ = clip.load("ViT-B/32", device=device)
		self.clip_model.eval()
		self.preprocess = transforms.Resize([224, 224])  # for cal loss

		# #######################generator and discriminators############## #
		netG = G_STYLE(
			size=self.args.size, 
			hidden_size=self.args.fc_dim, 
			style_dim=self.args.latent, 
			n_mlp=self.args.n_mlp,
			text_dim=self.args.text_dim,
			activation=self.args.activation, 
			mask_dim = 128,
			channel_multiplier=self.args.channel_multiplier,
		).to(device)
		

		n_scales = int(math.log(self.args.size//self.args.crop, 2)) + 1

		netD = D_NET(
			size=self.args.crop, 
			channel_multiplier=self.args.channel_multiplier, 
			n_scales=n_scales, 
			input_size=6,
			n_first_layers=self.args.n_first_layers,
		).to(device)

		
		netG_ema = G_STYLE(
			size=self.args.size, 
			hidden_size=self.args.fc_dim, 
			style_dim=self.args.latent, 
			n_mlp=self.args.n_mlp,
			text_dim=self.args.text_dim,
			activation=self.args.activation, 
			mask_dim = 128,
			channel_multiplier=self.args.channel_multiplier,
		).to(device)
		netG_ema.eval()
		self.accumulate(netG_ema, netG, 0)
		
		if get_rank() == 0:
			print('init model complete')
			print('G\'s trainable parameters =', count_parameters(netG))
			print('D\'s trainable parameters =', count_parameters(netD))

		g_reg_ratio = cfg.TRAIN.G_REG_EVERY / (cfg.TRAIN.G_REG_EVERY + 1)
		d_reg_ratio = cfg.TRAIN.D_REG_EVERY / (cfg.TRAIN.D_REG_EVERY + 1)
		
		optimG = optim.Adam(
			netG.parameters(),
			lr=cfg.TRAIN.GENERATOR_LR * g_reg_ratio,
			betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
		)
		optimD = optim.Adam(
			netD.parameters(), 
			lr=cfg.TRAIN.DISCRIMINATOR_LR * d_reg_ratio,
			betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
		)

		epoch = 0
		if cfg.TRAIN.NET_G != '':
			Gname = cfg.TRAIN.NET_G
			istart = Gname.rfind('_') + 1
			iend = Gname.rfind('.')
			epoch = int(Gname[istart:iend]) + 1

			ckpt = torch.load(
				Gname, 
				map_location=lambda storage, loc: storage
			)
			print("load",Gname)
			netG.load_state_dict(ckpt["g"])
			netG_ema.load_state_dict(ckpt["g_ema"])
			netD.load_state_dict(ckpt["d"])
			optimG.load_state_dict(ckpt['g_optim'])
			optimD.load_state_dict(ckpt['d_optim'])
			
			if get_rank() == 0:
				print("load model:", Gname)

		if get_rank() == 0:
			print("distribute:", self.args.distributed)
		if self.args.distributed:
			netG = nn.parallel.DistributedDataParallel(
				netG, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
			)
			print(f"distributed G")
			netD = nn.parallel.DistributedDataParallel(
				netD, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
			)
			print(f"distributed D")

		if get_rank() == 0:
			print("model complete")

		return [netG, netD, netG_ema, optimG, optimD, epoch]


	def prepare_labels(self):
		batch_size = self.batch_size
		device = self.args.device
		real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))  # (N,)
		fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))  # (N,)
		match_labels = Variable(torch.LongTensor(range(batch_size)))    # [0,1,...,9]
		real_labels = real_labels.to(device)
		fake_labels = fake_labels.to(device)
		match_labels = match_labels.to(device)

		return real_labels, fake_labels, match_labels

	def save_model(self, g_module, d_module, g_ema, g_optim, d_optim, s_name):
		torch.save(
			{
				"g": g_module.state_dict(),
				"d": d_module.state_dict(),
				"g_ema": g_ema.state_dict(),
				"g_optim": g_optim.state_dict(),
				"d_optim": d_optim.state_dict(),
			}, 
			s_name,
		)


	def adjust_dynamic_range(self, data, drange_in, drange_out):
		if drange_in != drange_out:
			scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
			bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
			data = data * scale + bias
		return data


	def convert_to_img(self, im, drange=[0, 1]):
		assert im.ndim == 2 or im.ndim == 3
		if im.ndim == 3:
			if im.shape[0] == 1:
				im = im[0] # grayscale CHW => HW
			else:
				im = im.transpose(1, 2, 0) # CHW -> HWC
		# [-1, 1] --> [0, 255]
		# im = (im + 1.0) * 127.5
		# im = im.astype(np.uint8)
		im = self.adjust_dynamic_range(im, drange, [0,255])
		im = np.rint(im).clip(0, 255).astype(np.uint8)
		return Image.fromarray(im)
	

	def make_noise(self, batch, latent_dim, device):
		return torch.randn(batch, latent_dim, device=device)


	def save_grid_captions(self, grid_cap, filename):
		print("Saving real captions")
		n_sample = len(grid_cap)
		save_caps = []
		for i in range(n_sample):
			cap = [
				self.ixtoword[_].encode('ascii', 'ignore').decode('ascii') 
				for _ in grid_cap[i].data.cpu().numpy()
			]
			save_caps.append(' '.join(cap).replace('END','') + '\n\n')

		fullpath = os.path.join(self.image_dir, filename)
		with open(fullpath, 'w') as f:
			f.writelines(save_caps)


	def save_grid_images(self, images, filename):
		n_sample = images.size(0)
		utils.save_image(
			images,
			f"{self.image_dir}/{filename}",
			nrow=int(n_sample ** 0.5),
			normalize=True,
			range=(-1, 1),
		)


	def save_sample(self, split='train'):
		n_sample = self.args.n_sample
		dataset = self.data_set if split == 'train' else self.val_set
		
		samples = dataset.get_grid_data(n_sample)
		
		imgs,  mask ,caps, caplens, _, _ = prepare_data(samples) 
		
		word = None
		texts = self.get_text_input(caps)
		word = self.clip_word_emb(texts,self.clip_model).detach()
		sent = self.clip_model.encode_text(texts).float()
		sent = sent.detach()
		
		self.save_grid_images(imgs, f'real_{split}.png')
		self.save_grid_images(mask[2], f'real_mask_{split}.png')
		self.save_grid_captions(caps, f'real_{split}_caps.txt')

		return word, sent , mask



	def get_coords(self,hight,weight,integer_values):
		coords = tt.convert_to_coord_format(1, hight, weight, integer_values=integer_values)
		return coords

	def get_text_input(self, caps):
		device = self.args.device
		texts = []
		for cap in caps:
			text = ' '.join([self.ixtoword[idx.item()] for idx in cap])
			text = text.replace('END', '').strip()
			texts.append(text)
		texts = clip.tokenize(texts).to(device)
		return texts
	def get_text(self, caps):
		texts = []
		for cap in caps:
			text = ' '.join([self.ixtoword[idx.item()] for idx in cap])
			text = text.replace('END', '').strip()
			texts.append(text)
		return texts
	def clip_word_emb(self,text,clip_model):
		x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
		x = x + clip_model.positional_embedding.type(clip_model.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = clip_model.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = clip_model.ln_final(x).type(clip_model.dtype)
		sent_len = min(30,max(text.argmax(dim=-1)))
		y = x[:,:sent_len,:]
		return y.permute(0,2,1)
	def train(self):
		device = self.args.device
		batch_size = self.batch_size
		
		train_loader = self.sample_data(self.data_loader)
		path_loader = self.sample_data(self.path_loader)

		netG, netD, netG_ema, optimG, optimD, start_epoch = self.build_models()
		real_labels, fake_labels, match_labels = self.prepare_labels()
		# # (N,), (N,), [0,1,...,N]
		self.clip_loss = CLIPLoss(self.clip_model)

		mean_path_length = 0
		mean_path_length_avg = 0

		d_loss_val = 0
		g_loss_val = 0
		r1_loss = torch.tensor(0.0, device=device)
		path_loss = torch.tensor(0.0, device=device)
		path_lengths = torch.tensor(0.0, device=device)
		loss_dict = {}

		if self.args.distributed:
			g_module = netG.module
			d_module = netD.module 
		else:
			g_module = netG 
			d_module = netD 
		g_ema = netG_ema

		accum = 0.5 ** (32 / (10 * 1000))

		if get_rank() == 0:
			train_words, train_sent ,train_mask = self.save_sample('train')
			val_words, val_sent ,val_mask= self.save_sample('val')
			gen_words, gen_sent ,gen_mask= self.save_sample('gen')


		gen_iters = 0
		best_fid, best_ep = None, None
		self.epoch=start_epoch
		self.max_epoch = 1000
		w_clip = 0.03
		for epoch in range(start_epoch, self.max_epoch):
			self.epoch = epoch
			if self.args.distributed:
				self.data_sampler.set_epoch(epoch)

			start_t = time.time()
			elapsed = 0
			step = 0
			w_clip = min(1.0,0.03+epoch*0.005)
			#self.num_batches
			while step < self.num_batches:
				start_step = start_t = time.time()
				######################################################
				# (1) Prepare training data and Compute text embeddings
				######################################################
				data = next(train_loader)
				real_img, mask ,caps, _, _, _ = prepare_data(data)

				# get text input
				texts = self.get_text_input(caps)
				word_emb = self.clip_word_emb(texts,self.clip_model).detach()
				states = self.clip_model.encode_text(texts).float()
				states = states.detach()

				#######################################################
				# (2) Update D network
				######################################################
				self.requires_grad(g_module, False)
				self.requires_grad(d_module, True)
				noise = [torch.randn(self.args.batch, self.args.latent, device=device)]
				fake_img, _, _ = g_module(mask,word_emb, noise=noise,return_all_images = True)


				real_input = torch.cat((real_img,mask[2]),1)
				fake_input = torch.cat((fake_img,mask[2]),1) 

				loss_d, real_pred, fake_pred = d_logistic_loss(
					d_module, real_input, fake_input, states, real_labels, fake_labels
				)

				loss_dict["d"] = loss_d
				loss_dict["real_score"] = real_pred.mean()
				loss_dict["fake_score"] = fake_pred.mean()
				
				# backward and update parameters
				d_module.zero_grad()
				loss_d.backward()
				optimD.step()

				d_reg_every = cfg.TRAIN.D_REG_EVERY
				r1 = cfg.TRAIN.R1
				d_regularize = gen_iters % d_reg_every == 0
				
				if d_regularize:
					real_input.requires_grad = True
					r1_loss, real_pred = d_r1_loss(d_module, real_input, states)

					d_module.zero_grad()
					(r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()

					optimD.step()
				
				loss_dict["r1"] = r1_loss

				#######################################################
				# (3) Update G network: maximize log(D(G(z)))
				######################################################
				self.requires_grad(g_module, True)
				self.requires_grad(d_module, False)
				
				# noise.data.normal_(0, 1)  # (N,Z_DIM)
				noise=[torch.randn(self.args.batch, self.args.latent, device=device)]
				fake_img, _, _ = g_module(mask,word_emb,noise=noise,sentence=states,return_all_images = True)
				fake_input = torch.cat((fake_img,mask[2]),1) 

				loss_g,_,_ = pixel_g_nonsaturating_loss(
					d_module,real_input, fake_input, states, real_labels,
				)
				loss_clip = self.clip_loss(fake_img, texts, match_labels)
				loss_dict["clip"] = loss_clip
				loss_dict["g"] = loss_g 

				loss_total = loss_g + loss_clip
				# backward and update parameters
				g_module.zero_grad()
				loss_total.backward()
				optimG.step()

				g_reg_every = cfg.TRAIN.G_REG_EVERY
				path_regularzie = cfg.TRAIN.PATH_REGULARIZE
				g_regularize = gen_iters % g_reg_every == 0
				
				if g_regularize:
					pl_data = next(path_loader)
					_, pl_mask , pl_caps, pl_cap_lens, _, _ = prepare_data(pl_data)
					
					path_batch = self.path_batch
					pl_caps = pl_caps[:path_batch]
					pl_cap_lens = pl_cap_lens[:path_batch]

					pl_texts = self.get_text_input(pl_caps)
					pl_states = self.clip_model.encode_text(pl_texts).float()
					pl_states = pl_states.detach()
					pl_clip_word_emb = self.clip_word_emb(pl_texts,self.clip_model).detach()
					noise=[torch.randn(self.path_batch, self.args.latent, device=device)]
					pl_fake_img,  pl_dlatents, _ = \
						g_module(pl_mask,pl_clip_word_emb,noise=noise,return_latents=True)

					path_loss, mean_path_length, path_lengths = g_path_regularize(
						pl_fake_img, pl_dlatents, mean_path_length
					)

					g_module.zero_grad()
					weighted_path_loss = path_regularzie * g_reg_every * path_loss

					if self.path_batch_shrink: 
						weighted_path_loss += 0 * pl_fake_img[0, 0, 0, 0]   # ??

					weighted_path_loss.backward()
					optimG.step()

					mean_path_length_avg = (
						reduce_sum(mean_path_length).item() / get_world_size()
					)

				loss_dict["path"] = path_loss
				loss_dict["path_length"] = path_lengths.mean()

				self.accumulate(g_ema, g_module, accum)

				loss_reduced = reduce_loss_dict(loss_dict)

				d_loss_val = loss_reduced["d"].mean().item()
				g_loss_val = loss_reduced["g"].mean().item()
				clip_loss_val = loss_reduced["clip"].mean().item()
				r1_val = loss_reduced["r1"].mean().item()
				path_loss_val = loss_reduced["path"].mean().item()
				real_score_val = loss_reduced["real_score"].mean().item()
				fake_score_val = loss_reduced["fake_score"].mean().item()
				path_length_val = loss_reduced["path_length"].mean().item()

				elapsed += (time.time() - start_step)
				display_gap = 100
				if get_rank() == 0:
					if gen_iters % display_gap == 0:  # 100
						print(
							f'Epoch [{epoch}/{self.max_epoch}] '
							f'Step [{step}/{self.num_batches}] '
							f'Time [{elapsed/display_gap:.2f}s]'
						)
						elapsed = 0
						print(
							f"d: {d_loss_val:.4f}; "
							f"g: {g_loss_val:.4f}; "
							f"clip: {clip_loss_val:.4f}; "
							f"clip_weight: {w_clip:.4f}; "
							f"r1: {r1_val:.4f}; "
							f"path: {path_loss_val:.4f}; "
							f"mean path: {mean_path_length_avg:.4f}; "
							f"real_score_val: {real_score_val:.4f}; "
							f"fake_score_val: {fake_score_val:.4f}; "
						)
						print('-' * 40)
					
					log_info = {
						"Generator": g_loss_val,
						"CLIP": clip_loss_val,
						"Discriminator": d_loss_val,
						"R1": r1_val,
						"Path Length Regularization": path_loss_val,
						"Mean Path Length": mean_path_length,
						"Real Score": real_score_val,
						"Fake Score": fake_score_val,
						"Path Length": path_length_val,
					}
					for key, value in log_info.items():
						self.writer.add_scalar(f'loss/{key}', float(value), gen_iters)
						

				step += 1
				gen_iters += 1
	
			end_t = time.time()
			if get_rank() == 0:
				with torch.no_grad():
					g_ema.eval()
					print("Saving fake images for epoch%d..." % (epoch))
					noise=[torch.randn(self.args.n_sample, self.args.latent, device=device)]
					gen_sample, _, _ = g_ema(gen_mask,gen_words,noise=noise)
					self.save_grid_images(
						gen_sample, 
						f"fake_{str(epoch).zfill(4)}_gen.png"
					)
				if epoch%10 == 0:
					self.save_model(
							g_module, d_module, g_ema, optimG, optimD, 
							f"{self.model_dir}/ckpt_{str(epoch).zfill(4)}.pth"
						)
			if epoch<70 or epoch%10 !=0:
				continue
			if get_rank() == 0:
				print(cfg.CONFIG_NAME)
				print('''[%d/%d] Loss_D: %.4f Loss_G: %.4f Time: %.2fs''' % (
					epoch, self.max_epoch, d_loss_val, g_loss_val, end_t - start_t))
				with torch.no_grad():
					g_ema.eval()
					noise=[torch.randn(self.args.n_sample, self.args.latent, device=device)]
					train_sample, _, _ = g_ema(train_mask,train_words,noise=noise)
					noise=[torch.randn(self.args.n_sample, self.args.latent, device=device)]
					val_sample, _, _ = g_ema(val_mask,val_words,noise=noise)

					print("Saving fake images for epoch%d..." % (epoch))
					self.save_grid_images(
						train_sample, 
						f"fake_{str(epoch).zfill(4)}_train.png"
					)
					self.save_grid_images(
						val_sample, 
						f"fake_{str(epoch).zfill(4)}_val.png"
					)
					
					fid1, fid2 = self.eval1(g_ema)
					print(fid1,"   ",fid2)
					if best_fid is None or fid2 < best_fid:
						best_fid, best_ep = fid2, epoch

						self.save_model(
							g_module, d_module, g_ema, optimG, optimD, 
							f"{self.model_dir}/ckpt_best.pth"
						)

					self.writer.add_scalar(f'fid_train', float(fid1), epoch)
					self.writer.add_scalar(f'fid_val', float(fid2), epoch)
					print(
						f"FID(train/val): {fid1:.4f} / {fid2:.4f}, "
						f"best FID: {best_fid:.4f} at ep{best_ep}"
					)

					metric_file = os.path.join(self.out_dir, f'fid10k.txt')
					with open(metric_file, 'a') as f:
						f.write(
							f'epoch-{str(epoch).zfill(4)}\t\t'
							f'fid3k2 {fid1:.4f} / {fid2:.4f}\n'
						)
				
				print('-' * 89)

				if epoch % self.snapshot_interval == 0 or epoch == self.max_epoch:
					print('Saving models...')
					self.save_model(
						g_module, d_module, g_ema, optimG, optimD, 
						f"{self.model_dir}/ckpt_{str(epoch).zfill(4)}.pth"
					)


	def eval1(self, netG):
		batch_size = self.batch_size
		n_batch = self.args.n_val // batch_size
		act = []
		data_iter = iter(self.gen_loader)
		self.fid_save_path = os.path.join(self.args.path_fid, 'fid_epoch'+str(self.epoch))
		fid_train,fid_val = calculate_fid(netG, val_dataset=self.eval_val_set,train_dataset = self.eval_data_set, bs=self.batch_size, textEnc = self.clip_model,
														num_batches=self.args.n_val // batch_size, latent_size=self.args.latent,get_text_input=self.get_text_input,
														save_dir=self.fid_save_path, data_iter=data_iter,prepare_data =prepare_data,val_loader=self.gen_loader,
														get_text = self.get_text,word2id = self.word2id,clip_word_emb = self.clip_word_emb)
		return fid_train['frechet_inception_distance'], fid_val['frechet_inception_distance']
