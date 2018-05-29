import torch, os
import numpy as np
from torch import optim
from torch import  nn
from MiniImagenet import MiniImagenet
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
	argparser.add_argument('-b', help='batch size', default=4)
	argparser.add_argument('-l', help='meta learning rate', default=1e-3)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	meta_batchsz = int(args.b)
	meta_lr = float(args.l)
	train_lr = 1e-2

	k_query = 15
	imgsz = 84
	mdl_file = 'ckpt/miniimagenet%d%d.mdl'%(n_way, k_shot)
	print('mini-imagnet: %d-way %d-shot meta-lr:%f, train-lr:%f' % (n_way, k_shot, meta_lr, train_lr))



	device = torch.device('cuda:0')
	net = MAML(n_way, k_shot, k_query, meta_batchsz, 5, meta_lr, train_lr, device)
	print(net)



	for epoch in range(1000):
		# batchsz here means total episode number
		mini = MiniImagenet('/hdd1/liangqu/datasets/miniimagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=imgsz)
		# fetch meta_batchsz num of episode each time
		db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=4, pin_memory=True)

		for step, batch in enumerate(db):

			# 2. train
			support_x = batch[0].to(device)
			support_y = batch[1].to(device)
			query_x = batch[2].to(device)
			query_y = batch[3].to(device)

			accs = net(support_x, support_y, query_x, query_y, training = True)

			if step % 50 == 0:
				print(epoch, step, '\t', accs)







if __name__ == '__main__':
	main()
