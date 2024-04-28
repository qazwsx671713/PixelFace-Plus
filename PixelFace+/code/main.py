from __future__ import print_function

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np

from ipdb import set_trace
from tqdm import tqdm

import torch
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer
from distributed import (
	get_rank, 
	synchronize, 
	cleanup_distributed
)
import sys
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
	parser = argparse.ArgumentParser(description="trainer")

	parser.add_argument(
		'--cfg', 
		type=str,
		default='cfg/face_v1.0_styleG.yml',  
		dest='cfg_file', 
		help='optional config file', 
	)
	parser.add_argument(
		'--data_dir', type=str, default='', dest='data_dir', 
	)
	parser.add_argument(
		'--NET_G', type=str, default=''
	)
	parser.add_argument(
		'--manualSeed', type=int, default=3201, help='manual seed'
	)
	parser.add_argument(
		'--local_rank', type=int, default=0, help='local rank for distributed training'
	)
	parser.add_argument('--distribute', type=bool,default=False)
	parser.add_argument(
		"--n_sample",
		type=int,
		default=9,
		help="number of the samples generated during training",
	)
	parser.add_argument(
		"--n_val",
		type=int,
		default=30000,
		help="number of the samples generated during eval",
	)
	parser.add_argument('--size', type=int, default=256)
	parser.add_argument('--fc_dim', type=int, default=512)
	parser.add_argument('--latent', type=int, default=512)
	parser.add_argument('--activation', type=str, default=None)
	parser.add_argument('--channel_multiplier', type=int, default=2)
	parser.add_argument('--coords_integer_values', action='store_true')
	parser.add_argument('--coords_size', type=int, default=256)
	parser.add_argument('--crop', type=int, default=256)
	parser.add_argument('--n_first_layers', type=int, default=0)
	parser.add_argument('--path_fid', type=str, default=None)
	args = parser.parse_args()
	return args


def data_sampler(dataset, shuffle, distributed):
	if distributed:
		return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

	elif shuffle:
		return data.RandomSampler(dataset)

	else:
		return data.SequentialSampler(dataset)


if __name__ == "__main__":
	args = parse_args()
	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)

	gpus = os.environ["CUDA_VISIBLE_DEVICES"]
	gpus = [int(gpu) for gpu in gpus.split(',')]

	n_gpu = len(gpus)
	args.distributed = n_gpu > 1

	if get_rank() == 0:
		print("distribute:", args.distributed)


	if args.distributed:
		args.local_rank = int(os.environ['LOCAL_RANK'])
		rank = int(os.environ['RANK'])
	else:
		args.local_rank = 0
		rank = 0
	world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
	args.distributed = world_size > 1


	if n_gpu >= 1:
		args.device = 'cuda'
	else:
		args.device = 'cpu'
	print(args.device)
	if args.NET_G != '':
		cfg.TRAIN.NET_G = args.NET_G
	if args.data_dir != '':
		cfg.DATA_DIR = args.data_dir

	if not cfg.TRAIN.FLAG:
		args.manualSeed = 3201
	elif args.manualSeed is None:
		args.manualSeed = random.randint(1, 10000)

	random.seed(args.manualSeed)
	np.random.seed(args.manualSeed)
	torch.manual_seed(args.manualSeed)
	torch.cuda.manual_seed_all(args.manualSeed)

	if args.distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(
			backend="nccl", 
			init_method="env://",
			timeout=datetime.timedelta(seconds=3600),
			rank=rank, 
			world_size=world_size
		)
		synchronize()

	cudnn.benchmark = True
	cudnn.deterministic = True
	if get_rank() == 0:
		print("Seed: %d" % (args.manualSeed))

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y%m%d%H%M')

	output_dir = '../output/%s_%s_%s' \
		% (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
	args.path_fid = output_dir+'/fid'
	args.batch = cfg.TRAIN.BATCH_SIZE
	args.dis_input_size = 3
	args.text_dim = cfg.TEXT.EMBEDDING_DIM
	args.n_mlp = cfg.GAN.N_MLP

	algo = trainer(output_dir, args)

	start_t = time.time()
	if cfg.TRAIN.FLAG:  # True for training, False for generating images
		algo.train()
		cleanup_distributed()
	else:
		# todo sampling
		algo.sample()
		
	end_t = time.time()
	print(f'Total time: {(end_t - start_t):.4f}')

