import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np




class Net(nn.Module):
	"""
	This is the theta network structure.
	Unlike traditonal nn.Module, we need expose theta tensors and explicit forward function runing on 3nd weights,
	which makes this module a little bit redundant to read.
	Notice: Pytorch 0.4.0 +
	for F.conv2d, weights = [out ch, in ch, kernel h, kenel w]
	for F.linear, weights = [out dim, in dim]
	"""

	def __init__(self, nway):
		"""

		:param nway: N-way
		:param device: device environment
		"""
		super(Net, self).__init__()

		ch = 32

		# this dict contains all tensors needed to be optimized
		self.vars = nn.ParameterList([
			#'conv1w'        
			nn.Parameter(torch.empty(ch, 3, 3, 3)),
			#'conv1b'        
			nn.Parameter(torch.empty(ch)),
			#'conv1bn_w'     
			nn.Parameter(torch.empty(ch)),
			#'conv1bn_b'     
			nn.Parameter(torch.empty(ch)),

			# conv2:
			#'conv2w'        
			nn.Parameter(torch.empty(ch, ch, 3, 3)),
			#'conv2b'        
			nn.Parameter(torch.empty(ch)),
			#'conv2bn_w'     
			nn.Parameter(torch.empty(ch)),
			#'conv2bn_b'     
			nn.Parameter(torch.empty(ch)),

			# conv3:
			#'conv3w'        
			nn.Parameter(torch.empty(ch, ch, 3, 3)),
			#'conv3b'        
			nn.Parameter(torch.empty(ch)),
			#'conv3bn_w'     
			nn.Parameter(torch.empty(ch)),
			#'conv3bn_b'     
			nn.Parameter(torch.empty(ch)),

			# conv4:
			#'conv4w'        
			nn.Parameter(torch.empty(ch, ch, 3, 3)),
			#'conv4b'        
			nn.Parameter(torch.empty(ch)),
			#'conv4bn_w'     
			nn.Parameter(torch.empty(ch)),
			#'conv4bn_b'     
			nn.Parameter(torch.empty(ch)),

			# fc1:
			#'fc1w'
			nn.Parameter(torch.empty(nway, ch * 3 * 3)),
			#'fc1b'
			nn.Parameter(torch.empty(nway)),
		])

		# this dict contains global moving_mean/moving_variance for all batch norm layers.
		# NOTICE: it requires no gradients for batch norm moving mean and moving variance.
		self.bns = nn.ParameterList([
			# conv1 bn:
			#'conv1bn_mean'  
			nn.Parameter(torch.empty(ch), requires_grad=False),
			#'conv1bn_var'   
			nn.Parameter(torch.empty(ch), requires_grad=False),

			# conv2 bn:
			#'conv2bn_mean'  
			nn.Parameter(torch.empty(ch), requires_grad=False),
			#'conv2bn_var'   
			nn.Parameter(torch.empty(ch), requires_grad=False),

			# conv3 bn:
			#'conv3bn_mean'  
			nn.Parameter(torch.empty(ch), requires_grad=False),
			#'conv3bn_var'   
			nn.Parameter(torch.empty(ch), requires_grad=False),

			# conv4 bn:
			#'conv4bn_mean'  
			nn.Parameter(torch.empty(ch), requires_grad=False),
			#'conv4bn_var'   
			nn.Parameter(torch.empty(ch), requires_grad=False),
		])



	def weights_init(self):

		var_idx = bn_idx = 0

		# conv1
		nn.init.xavier_normal_(self.vars[var_idx])  # conv weight
		self.vars[var_idx + 1].data.zero_()         # conv bias
		self.vars[var_idx + 2].data.fill_(1)        # bn weight
		self.vars[var_idx + 3].data.zero_()         # bn bias
		self.bns[bn_idx].data.zero_()               # bn moving_mean TODO: 0 or 1
		self.bns[bn_idx + 1].data.zero_()           # bn moving_variance TODO: 0 or 1
		var_idx += 4
		bn_idx += 2

		# conv2
		nn.init.xavier_normal_(self.vars[var_idx])  # conv weight
		self.vars[var_idx + 1].data.zero_()         # conv bias
		self.vars[var_idx + 2].data.fill_(1)        # bn weight
		self.vars[var_idx + 3].data.zero_()         # bn bias
		self.bns[bn_idx].data.zero_()               # bn moving_mean
		self.bns[bn_idx + 1].data.zero_()           # bn moving_variance
		var_idx += 4
		bn_idx += 2

		# conv3
		nn.init.xavier_normal_(self.vars[var_idx])  # conv weight
		self.vars[var_idx + 1].data.zero_()         # conv bias
		self.vars[var_idx + 2].data.fill_(1)        # bn weight
		self.vars[var_idx + 3].data.zero_()         # bn bias
		self.bns[bn_idx].data.zero_()               # bn moving_mean
		self.bns[bn_idx + 1].data.zero_()           # bn moving_variance
		var_idx += 4
		bn_idx += 2

		# conv4
		nn.init.xavier_normal_(self.vars[var_idx])  # conv weight
		self.vars[var_idx + 1].data.zero_()         # conv bias
		self.vars[var_idx + 2].data.fill_(1)        # bn weight
		self.vars[var_idx + 3].data.zero_()         # bn bias
		self.bns[bn_idx].data.zero_()               # bn moving_mean
		self.bns[bn_idx + 1].data.zero_()           # bn moving_variance
		var_idx += 4
		bn_idx += 2

		# fc 1
		self.vars[var_idx].data.fill_(1)            # fc weight
		self.vars[var_idx + 1].data.zero_()         # fc bias





	def test(self):
		x = torch.randn(4, 3, 84, 84)
		x = self(x)



	def forward(self, x, vars=None, bns=None, training=True):
		"""

		:param x:   [sz, c_, h, w]
		:param vars: list of tensors
		:param bns: list of tensors
		:param training: train or test, for batch norm
		:return:
		"""
		if vars is None:
			vars = self.vars

		if bns is None:
			bns = self.bns

		var_idx = bn_idx = 0
		batchsz = x.size(0)

		# conv1
		# print('=conv1 x:', x.size())
		x = F.conv2d(x, vars[var_idx], vars[var_idx + 1], stride = 1, padding = 0)
		# print('conv', x.size())
		x = F.batch_norm(x, bns[bn_idx], bns[bn_idx + 1],
		                 weight=vars[var_idx + 2],
		                 bias=vars[var_idx + 3],
		                 training=training)
		# print('bn', x.size())
		x = F.relu(x, inplace=True)
		x = F.max_pool2d(x, kernel_size=2, padding=0)
		# print('pool', x.size())
		var_idx += 4
		bn_idx += 2

		# conv2
		# print('=conv2 x', x.size())
		x = F.conv2d(x, vars[var_idx], vars[var_idx + 1], stride = 1, padding = 0)
		# print('conv', x.size())
		x = F.batch_norm(x, bns[bn_idx], bns[bn_idx + 1],
		                 weight=vars[var_idx + 2],
		                 bias=vars[var_idx + 3],
		                 training=training)
		x = F.relu(x, inplace=True)
		x = F.max_pool2d(x, kernel_size=2, padding=0)
		# print('pool', x.size())
		var_idx += 4
		bn_idx += 2

		# conv3
		# print('=conv3 x', x.size())
		x = F.conv2d(x, vars[var_idx], vars[var_idx + 1], stride = 1, padding = 0)
		# print('conv', x.size())
		x = F.batch_norm(x, bns[bn_idx], bns[bn_idx + 1],
		                 weight=vars[var_idx + 2],
		                 bias=vars[var_idx + 3],
		                 training=training)
		x = F.relu(x, inplace=True)
		x = F.max_pool2d(x, kernel_size=2, padding=0)
		# print('pool', x.size())
		var_idx += 4
		bn_idx += 2

		# conv3
		# print('=conv4 x', x.size())
		x = F.conv2d(x, vars[var_idx], vars[var_idx + 1], stride = 1, padding = 0)
		# print('conv', x.size())
		x = F.batch_norm(x, bns[bn_idx], bns[bn_idx + 1],
		                 weight=vars[var_idx + 2],
		                 bias=vars[var_idx + 3],
		                 training=training)
		x = F.relu(x, inplace=True)
		x = F.max_pool2d(x, kernel_size=2, padding=0)
		# print('pool', x.size())
		var_idx += 4
		bn_idx += 2

		# fc1
		# print('=fc1 x', x.size())
		x = x.view(batchsz, -1)
		# print('flatten', x.size())
		x = F.linear(x, vars[var_idx], vars[var_idx + 1])
		var_idx += 2
		# print('fc', x.size())


		return x


	def zero_grad(self, vars=None):
		"""
		operate on 3nd weights or class weights
		clear all grad data.
		:param vars:
		:return:
		"""
		if vars is None:
			for p in self.vars:
				if p.grad is not None:
					p.grad.zero_()
		else:
			for p in vars:
				if p.grad is not None:
					p.grad.zero_()

	def parameters(self, vars=None):
		"""

		:param vars: list
		:return:
		"""
		if vars is None:
			return self.vars

		else:
			return vars

	def __repr__(self):
		return "MAML Basic Net(Variables:{0} Tensors:{1})".format(len(self.vars), len(self.bns))



class MAML(nn.Module):


	def __init__(self, nway, kshot, kquery, meta_batchsz, K, meta_lr, train_lr):
		"""

		:param nway:
		:param kshot:
		:param kquery:
		:param meta_batchsz: tasks number
		:param K:   inner update steps
		:param device:
		"""
		super(MAML, self).__init__()

		self.train_lr = train_lr
		self.meta_lr = meta_lr
		self.nway = nway
		self.kshot = kshot
		self.kquery = kquery
		self.meta_batchsz = meta_batchsz
		self.K = K


		self.net = Net(nway)
		self.net.weights_init()

		self.meta_optim = optim.Adam(self.net.parameters(), lr= self.meta_lr)

	def forward(self, support_x, support_y, query_x, query_y, training):
		"""

		:param support_x:   [b, setsz, c_, h, w]
		:param support_y:   [b, setsz]
		:param query_x:     [b, querysz, c_, h, w]
		:param query_y:     [b, querysz]
		:param training:
		:return:
		"""
		batchsz, setsz, c_, h, w = support_x.size()
		querysz = query_x.size(1)


		losses_q = [] # losses_q[i], i is tasks idx
		corrects = [0] * (self.K+1) # corrects[i] save cumulative correct number of all tasks in step k

		# TODO: add multi-threading support
		# NOTICE: although the GIL limit the multi-threading performance severely, it does make a difference. 
		# When deal with IO operation,
		# we need to coordinate with outside IO devices. With the assistance of multi-threading, we can issue multi-commands
		# parallelly and improve the efficency of IO usage.
		for i in range(batchsz): # batchsz==self.meta_batchsz

			# 1. run the i-th task and compute loss for k=0
			pred = self.net(support_x[i])
			loss = F.cross_entropy(pred, support_y[i])

			# 2. grad on theta
			# clear theta grad info
			self.net.zero_grad()
			grad = torch.autograd.grad(loss, self.net.parameters())

			# print('k0')
			# for p in grad[:5]:
			# 	print(p.norm().item())

			# 3. theta_pi = theta - train_lr * grad
			fast_weights = list(map(lambda p: p[1] - self.train_lr * p[0], zip(grad, self.net.parameters())))

			# this is the loss and accuracy before first update
			# [setsz, nway]
			pred_q = self.net(query_x[i], self.net.parameters(), bns=None, training=training)
			# [setsz]
			pred_q = F.softmax(pred_q, dim=1).argmax(dim=1)
			# scalar
			correct = torch.eq(pred_q, query_y[i]).sum().item()
			corrects[0] = corrects[0] + correct

			# this is the loss and accuracy after the first update
			# [setsz, nway]
			pred_q = self.net(query_x[i], fast_weights, bns=None, training=training)
			loss_q = F.cross_entropy(pred_q, query_y[i])
			# [setsz]
			pred_q = F.softmax(pred_q, dim=1).argmax(dim=1)
			# scalar
			correct = torch.eq(pred_q, query_y[i]).sum().item()
			corrects[1] = corrects[1] + correct

			for k in range(1, self.K):
				# 1. run the i-th task and compute loss for k=1~K-1
				pred = self.net(support_x[i], fast_weights, bns=None, training=training)
				loss = F.cross_entropy(pred, support_y[i])
				# clear fast_weights grad info
				self.net.zero_grad(fast_weights)
				# 2. compute grad on theta_pi
				grad = torch.autograd.grad(loss, fast_weights)
				# 3. theta_pi = theta_pi - train_lr * grad
				fast_weights = list(map(lambda p: p[1] - self.train_lr * p[0], zip(grad, fast_weights)))


				pred_q = self.net(query_x[i], fast_weights, bns=None, training=training)
				loss_q = F.cross_entropy(pred_q, query_y[i])
				pred_q = F.softmax(pred_q, dim=1).argmax(dim=1)
				correct = torch.eq(pred_q, query_y[i]).sum().item() # convert to numpy
				corrects[k+1] = corrects[k+1] + correct

			# 4. record last step's loss for task i
			losses_q.append(loss_q)

		# end of all tasks
		# sum over all losses across all tasks
		loss_q = torch.stack(losses_q).sum(0)
		
		if training: # update theta when training
			# optimize theta parameters
			self.meta_optim.zero_grad()
			loss_q.backward()
			# print('meta update')
			# for p in self.net.parameters()[:5]:
			# 	print(torch.norm(p).item())
			self.meta_optim.step()

		accs = np.array(corrects) / (querysz * batchsz)

		return accs











def test():
	net = Net(5)
	net.test()




if __name__ == '__main__':
	test()






















