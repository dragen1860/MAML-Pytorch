import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from MiniImagenet import MiniImagenet
from naive5 import Naive5
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
from torch import nn

global_train_acc_buff = 0
global_train_loss_buff = 0
global_test_acc_buff = 0
global_test_loss_buff = 0
global_buff = []


def write2file(n_way, k_shot):
    global_buff.append([global_train_loss_buff, global_train_acc_buff, global_test_loss_buff, global_test_acc_buff])
    with open("mini%d%d.pkl" % (n_way, k_shot), "wb") as fp:
        pickle.dump(global_buff, fp)


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


# save best acc info, to save the best model to ckpt.
best_accuracy = 0


def evaluation(net, batchsz, n_way, k_shot, imgsz, episodesz, threhold, mdl_file):
    """
    obey the expriment setting of MAML and Learning2Compare, we randomly sample 600 episodes and 15 query images per query
    set.
    :param net:
    :param batchsz:
    :return:
    """
    k_query = 15
    mini_val = MiniImagenet('../mini-imagenet/', mode='test', n_way=n_way, k_shot=k_shot, k_query=k_query,
                            batchsz=600, resize=imgsz)
    db_val = DataLoader(mini_val, batchsz, shuffle=True, num_workers=2, pin_memory=True)

    accs = []
    episode_num = 0  # record tested num of episodes

    for batch_test in db_val:
        # [60, setsz, c_, h, w]
        # setsz = (5 + 15) * 5
        support_x = Variable(batch_test[0]).cuda()
        support_y = Variable(batch_test[1]).cuda()
        query_x = Variable(batch_test[2]).cuda()
        query_y = Variable(batch_test[3]).cuda()

        # we will split query set into 15 splits.
        # query_x : [batch, 15*way, c_, h, w]
        # query_x_b : tuple, 15 * [b, way, c_, h, w]
        query_x_b = torch.chunk(query_x, k_query, dim=1)
        # query_y : [batch, 15*way]
        # query_y_b: 15* [b, way]
        query_y_b = torch.chunk(query_y, k_query, dim=1)
        preds = []
        net.eval()
        # we don't need the total acc on 600 episodes, but we need the acc per sets of 15*nway setsz.
        total_correct = 0
        total_num = 0
        total_loss = 0
        for query_x_mini, query_y_mini in zip(query_x_b, query_y_b):
            # print('query_x_mini', query_x_mini.size(), 'query_y_mini', query_y_mini.size())
            loss, pred, correct = net(support_x, support_y, query_x_mini.contiguous(), query_y_mini, False)
            correct = correct.sum()  # multi-gpu
            # pred: [b, nway]
            preds.append(pred)
            total_correct += correct.data[0]
            total_num += query_y_mini.size(0) * query_y_mini.size(1)

            total_loss += loss.data[0]

        # # 15 * [b, nway] => [b, 15*nway]
        # preds = torch.cat(preds, dim= 1)
        acc = total_correct / total_num
        print('%.5f,' % acc, end=' ')
        sys.stdout.flush()
        accs.append(acc)

        # update tested episode number
        episode_num += query_y.size(0)
        if episode_num > episodesz:
            # test current tested episodes acc.
            acc = np.array(accs).mean()
            if acc >= threhold:
                # if current acc is very high, we conduct all 600 episodes testing.
                continue
            else:
                # current acc is low, just conduct `episodesz` num of episodes.
                break

    # compute the distribution of 600/episodesz episodes acc.
    global best_accuracy
    accs = np.array(accs)
    accuracy, sem = mean_confidence_interval(accs)
    print('\naccuracy:', accuracy, 'sem:', sem)
    print('<<<<<<<<< accuracy:', accuracy, 'best accuracy:', best_accuracy, '>>>>>>>>')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(net.state_dict(), mdl_file)
        print('Saved to checkpoint:', mdl_file)

    # we only take the last one batch as avg_loss
    total_loss = total_loss / n_way / k_query

    global global_test_loss_buff, global_test_acc_buff
    global_test_loss_buff = total_loss
    global_test_acc_buff = accuracy
    write2file(n_way, k_shot)

    return accuracy, sem


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', help='n way')
    argparser.add_argument('-k', help='k shot')
    argparser.add_argument('-b', help='batch size')
    argparser.add_argument('-l', help='learning rate', default=1e-3)
    args = argparser.parse_args()
    n_way = int(args.n)
    k_shot = int(args.k)
    batchsz = int(args.b)
    lr = float(args.l)

    k_query = 1
    imgsz = 224
    threhold = 0.699 if k_shot == 5 else 0.584  # threshold for when to test full version of episode
    mdl_file = 'ckpt/naive5_3x3%d%d.mdl' % (n_way, k_shot)
    print('mini-imagnet: %d-way %d-shot lr:%f, threshold:%f' % (n_way, k_shot, lr, threhold))

    global global_buff
    if os.path.exists('mini%d%d.pkl' % (n_way, k_shot)):
        global_buff = pickle.load(open('mini%d%d.pkl' % (n_way, k_shot), 'rb'))
        print('load pkl buff:', len(global_buff))

    net = nn.DataParallel(Naive5(n_way, k_shot, imgsz), device_ids=[0, 1, 2]).cuda()
    print(net)

    if os.path.exists(mdl_file):
        print('load from checkpoint ...', mdl_file)
        net.load_state_dict(torch.load(mdl_file))
    else:
        print('training from scratch.')

    # whole parameters number
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total params:', params)

    # build optimizer and lr scheduler
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=25, verbose=True)

    for epoch in range(1000):
        mini = MiniImagenet('../mini-imagenet/', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query,
                            batchsz=10000, resize=imgsz)
        db = DataLoader(mini, batchsz, shuffle=True, num_workers=8, pin_memory=True)
        total_train_loss = 0
        total_train_correct = 0
        total_train_num = 0

        for step, batch in enumerate(db):
            # 1. test
            if step % 300 == 0:
                # evaluation(net, batchsz, n_way, k_shot, imgsz, episodesz, threhold, mdl_file):
                accuracy, sem = evaluation(net, batchsz, n_way, k_shot, imgsz, 600, threhold, mdl_file)
                scheduler.step(accuracy)

            # 2. train
            support_x = Variable(batch[0]).cuda()
            support_y = Variable(batch[1]).cuda()
            query_x = Variable(batch[2]).cuda()
            query_y = Variable(batch[3]).cuda()

            net.train()
            loss, pred, correct = net(support_x, support_y, query_x, query_y)
            loss = loss.sum() / support_x.size(0)  # multi-gpu, divide by total batchsz
            total_train_loss += loss.data[0]
            total_train_correct += correct.data[0]
            total_train_num += support_y.size(0) * n_way  # k_query = 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3. print
            if step % 20 == 0 and step != 0:
                acc = total_train_correct / total_train_num
                total_train_correct = 0
                total_train_num = 0

                print('%d-way %d-shot %d batch> epoch:%d step:%d, loss:%.4f, train acc:%.4f' % (
                    n_way, k_shot, batchsz, epoch, step, total_train_loss, acc))
                total_train_loss = 0

                global global_train_loss_buff, global_train_acc_buff
                global_train_loss_buff = loss.data[0] / (n_way * k_shot)
                global_train_acc_buff = acc
                write2file(n_way, k_shot)


if __name__ == '__main__':
    main()
