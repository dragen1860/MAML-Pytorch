import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse

from maml import MAML





def mean_confidence_interval(accs, confidence = 0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf( (1 + confidence) / 2, n - 1)
    return m, h

def main():
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-n', help='n way', default=5)
	argparser.add_argument('-k', help='k shot', default=1)
	argparser.add_argument('-b', help='meta batch size', default=4)
	argparser.add_argument('-l', help='meta learning rate', default=1e-3)
	args = argparser.parse_args()
	n_way = int(args.n)
	k_shot = int(args.k)
	meta_batchsz = int(args.b)
	meta_lr = float(args.l)

	train_lr = 1e-2
	k_query = 15
	imgsz = 84
	K = 5 # update steps
	mdl_file = 'ckpt/miniimagenet%d%d.mdl'%(n_way, k_shot)
	print('mini-imagenet: %d-way %d-shot meta-lr:%f, train-lr:%f K-steps:%d' % (n_way, k_shot, meta_lr, train_lr, K))



	device = torch.device('cuda:0')
	net = MAML(n_way, k_shot, k_query, meta_batchsz, K, meta_lr, train_lr).to(device) 
	print(net)



	for epoch in range(1000):
		# batchsz here means total episode number
		mini = MiniImagenet('/hdd1/liangqu/datasets/miniimagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
		                    batchsz=10000, resize=imgsz)
		# fetch meta_batchsz num of episode each time
		db = DataLoader(mini, meta_batchsz, shuffle=True, num_workers=meta_batchsz, pin_memory=True)

		for step, batch in enumerate(db):

			support_x = batch[0].to(device)
			support_y = batch[1].to(device)
			query_x = batch[2].to(device)
			query_y = batch[3].to(device)

			accs = net(support_x, support_y, query_x, query_y, training = True)

			if step % 50 == 0:
				print(epoch, step, '\t', accs)


			if step % 500 == 0 and step != 0: # evaluation
				# test for 600 episodes
				mini_test = MiniImagenet('/hdd1/liangqu/datasets/miniimagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
				                    batchsz=600, resize=imgsz)
				db_test = DataLoader(mini_test, meta_batchsz, shuffle=True, num_workers=meta_batchsz, pin_memory=True)
				accs_all_test = []
				for batch in db_test:
					support_x = batch[0].to(device)
					support_y = batch[1].to(device)
					query_x = batch[2].to(device)
					query_y = batch[3].to(device)

					accs = net(support_x, support_y, query_x, query_y, training = False)
					accs_all_test.append(accs)
				# [600, K+1]
				accs_all_test = np.array(accs_all_test)
				# [600, K+1] => [K+1]
				means = accs_all_test.mean(axis=0)
				# compute variance for last step K
				m, h = mean_confidence_interval(accs_all_test[:, K])
				print('>>Test:\t', means, 'variance[K]: %.4f'%h, '<<')







if __name__ == '__main__':
	main()
