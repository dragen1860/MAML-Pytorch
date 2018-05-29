import torch, os
import numpy as np
from torch import optim
from torch import  nn
from omniglotNShot import OmniglotNShot
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse

from maml import MAML



def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way', default=5)
	argparser.add_argument('-k', help='k shot', default=1)
	argparser.add_argument('-b', help='batch size', default=32)
	argparser.add_argument('-l', help='meta learning rate', default=1e-3)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	meta_batchsz = int(args.b)
	meta_lr = float(args.l)
	train_lr = 0.4

	k_query = 15
	imgsz = 84
	mdl_file = 'ckpt/omniglot%d%d.mdl'%(n_way, k_shot)
	print('omniglot: %d-way %d-shot meta-lr:%f, train-lr:%f' % (n_way, k_shot, meta_lr, train_lr))



	device = torch.device('cuda:0')
	net = MAML(n_way, k_shot, k_query, meta_batchsz, 1, meta_lr, train_lr, device)
	print(net)


	# batchsz here means total episode number
	db = OmniglotNShot('omniglot', batchsz=meta_batchsz, n_way=n_way, k_shot=k_shot, k_query=k_query, imgsz=imgsz)

	for step in range(10000000):

		# train
		support_x, support_y, query_x, query_y = db.get_batch('train')
		support_x = torch.from_numpy(support_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1).to(device)
		query_x = torch.from_numpy(query_x).float().transpose(2, 4).transpose(3, 4).repeat(1, 1, 3, 1, 1).to(device)
		support_y = torch.from_numpy(support_y).long().to(device)
		query_y = torch.from_numpy(query_y).long().to(device)

		accs = net(support_x, support_y, query_x, query_y, training = True)

		if step % 50 == 0:
			print(step, '\t', accs)



		if step % 1000 == 0:
			# test
			pass





if __name__ == '__main__':
	main()
