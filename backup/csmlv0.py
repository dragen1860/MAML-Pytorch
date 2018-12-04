import torch
from torch import nn
from torch import optim
from torch import autograd
from torch import multiprocessing
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import TensorDataset
import numpy as np
import os

multiprocessing = multiprocessing.get_context('spawn')


class Concept(nn.Module):

    def __init__(self):
        super(Concept, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(inplace=True),
        )

    def load(self, src):
        """
        Load parameters from central pool
        :param src:
        :return:
        """
        self.load_state_dict(src.state_dict())

    def forward(self, x):
        x = self.net(x)

        return x


class Relation(nn.Module):

    def __init__(self):
        super(Relation, self).__init__()

        self.g = nn.Sequential(
            nn.Linear(2 * (64 + 2), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        self.f = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        pass


class OutLayer(nn.Module):

    def __init__(self):
        super(OutLayer, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(64 * 3 * 3, 5)
        )

    def forward(self, x):
        # downsample
        x = F.avg_pool2d(x, 5, 5)
        # flatten
        x = x.view(x.size(0), -1)
        # print(x.size())
        return self.net(x)


def inner_train(K, gpuidx, support_x, support_y, query_x, query_y, concepts, Q):
    """
    inner-loop train function.
    :param K: train iterations
    :param gpuidx: which gpu to train
    :param support_x:   [b, setsz, c_, h, w]
    :param support_y:   []
    :param query_x:     [b, querysz]
    :param query_y:
    :param concepts:    concepts network
    :param Q:           Queue to receive result
    :return:
    """
    #
    assert support_x.size(0) == query_x.size(0)
    # move current tensor into working GPU card.
    support_x = support_x.cuda(gpuidx)
    support_y = support_y.cuda(gpuidx)
    query_x = query_x.cuda(gpuidx)
    query_y = query_y.cuda(gpuidx)

    support_db = TensorDataset(support_x, support_y)
    query_db = TensorDataset(query_x, query_y)

    # this is inner-loop, update for K steps for one task.
    outlayer = OutLayer().cuda(gpuidx)
    criteon = nn.CrossEntropyLoss().cuda(gpuidx)
    # this is inner-loop optimizer, and corresponding lr stands for update-lr
    optimizer = optim.Adam(outlayer.parameters(), lr=1e-3)

    # right = [0] on gpuidx
    right = Variable(torch.zeros(1).cuda(gpuidx))
    # loss = [0] on gpuidx
    loss = Variable(torch.zeros(1).cuda(gpuidx))
    for (support_xb, support_yb), (query_xb, query_yb) in zip(support_db, query_db):
        # support_xb: [setsz, c_, h, w]
        # support_yb: [setsz]
        # query_xb  : [querysz, c_, h, w]
        # query_yb  : [querysz]
        # 1. meta-train for K iterations on meta-train dataset
        for i in range(K):
            # get the representation from concept-network
            x = concepts[gpuidx](support_xb)
            # detach gradient backpropagation
            x = x.detach()
            # push to outlayer-network
            logits = outlayer(x)
            # compute loss
            loss = criteon(logits, support_yb)

            # backward
            outlayer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2. meta-test on meta-test dataset
        x = concepts[gpuidx](query_xb)
        # [querysz, nway]  [querysz]
        logits = outlayer(x)
        _, idx = logits.max(1)
        # convert ByteTensor to LongTensor
        pred = idx.long()

        # 3. accumulate all right num and loss
        # torch.eq() return with ByteTensor
        # we use logits to compute loss while use pred to calculate accuracy
        right += torch.eq(pred, query_yb).sum().float()
        loss += criteon(logits, query_yb)

    # compute accuracy
    accuracy = right.data[0] / np.array(query_y.size()).prod()

    print(gpuidx, loss.data[0], accuracy)
    # save meta-test-loss into Queue for current task
    # just save data, not TENSOR
    Q.put([gpuidx, loss.data[0], accuracy])

    del outlayer, criteon
    print('removed outlayer and criteon.')


class CSML:
    """
    Concept-Sharing Meta-Learning
    """

    def __init__(self):

        # num of task training in parallel
        self.N = 3
        # inner-loop update iteration
        self.K = 10

        # each task has individual concept and output network, we deploy them on distinct GPUs and
        # merge into a list.
        self.concepts = []
        self.outlayers = []
        self.optimizer = None

        # to save async multi-tasks' loss and accuracy
        self.Q = multiprocessing.Queue()

        print('please call deploy() func to deploy networks. DO NOT call cuda() explicitly.')

    def deploy(self):
        # deplay N task on distributed GPU cluster and
        # append instance into list
        for i in range(self.N):
            concept = Concept().cuda(i)
            outlayer = OutLayer().cuda(i)
            self.concepts.append(concept)
            self.outlayers.append(outlayer)

        # meta optimizer
        self.optimizer = optim.Adam(self.concepts[0].parameters(), lr=1e-3)
        print('deploy done.')

    def train(self, support_x, support_y, query_x, query_y, train=True):
        """
        This is meta-train and meta-test function.
        :param support_x:   [batchsz, setsz, c_, h, w]
        :param support_y:   [batchsz, setsz]
        :param query_x:     [batchsz, querysz]
        :param query_y:     [batchsz, querysz]
        :return:
        """
        # we need split single batch into several batch to process asynchonuous
        batchsz = support_x.size(0)
        support_xb = torch.chunk(support_x, self.N)
        support_yb = torch.chunk(support_y, self.N)
        query_xb = torch.chunk(query_x, self.N)
        query_yb = torch.chunk(query_y, self.N)

        # 1. download latest Concept Network weights from central pool
        # here download from GPU0
        for i in range(1, self.N):
            self.concepts[i].load(self.concepts[0])

        # 3. start training for whole tasks asynchronous
        processes = []
        for p in range(self.N):
            p = multiprocessing.Process(target=inner_train,
                                        args=(self.K, p, support_xb[p], support_yb[p], query_xb[p], query_yb[p],
                                              self.concepts, self.Q))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        print('join completed.')
        # 4. merge result
        # util here, we have executed all tasks in GPU cluster in parallel.
        data = [self.Q.get_nowait() for _ in range(self.N)]
        accuracy = np.array([i[2] for i in data]).mean()
        meta_train_loss = np.array([i[1] for i in data]).astype(np.float32).sum()
        # meta_train_loss = Variable(torch.FloatTensor(meta_train_loss).cuda(0))

        print('acc:', accuracy, 'meta-loss:', meta_train_loss)

        if train:
            # compute gradients
            autograd.grad()
            dummy_x = support_x[0][:2].cuda(0)
            # update concept network.
            self.optimizer.zero_grad()
            # [2, c_, h, w]
            dummy_loss = self.concepts[0](dummy_x)
            dummy_loss.backward(torch.FloatTensor([1]).cuda(0))
            self.optimizer.step()
