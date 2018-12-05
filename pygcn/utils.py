import numpy as np
import scipy.sparse as sp
import torch
import time
import os.path as osp
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def train(epoch, net, data_loader, optimizer, criterion):
    losses = 0
    correct = 0
    miss = 0
    net.train()
    t0 = time.time()
    for batch_idx, (target, feat, adj) in enumerate(data_loader):
        target = target.cuda().squeeze(0).long()
        target = target.view(target.shape[0] * target.shape[0], -1).squeeze(1)
        feat = feat.cuda().squeeze(0).float()
        adj = adj.cuda().squeeze(0).float()
        output = net(feat, adj)
        pred = torch.argmax(output, 1)
        correct += pred.eq(target).sum().item()
        miss += target.shape[0] - pred.eq(target).sum().item()
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        if (batch_idx + 1) % 100 == 0:
            t1 = time.time()
            t_batch = t1 - t0
            t0 = time.time()
            print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_batch))
    return losses / len(data_loader), correct / (correct + miss)


def test(net, data_loader, criterion, ):
    losses = 0
    correct = 0
    miss = 0
    net.eval()
    t0 = time.time()
    for batch_idx, (target, feat, adj) in enumerate(data_loader):
        target = target.cuda().squeeze(0).long()
        target = target.view(target.shape[0] * target.shape[0], -1).squeeze(1)
        feat = feat.cuda().squeeze(0).float()
        adj = adj.cuda().squeeze(0).float()
        output = net(feat, adj)
        pred = torch.argmax(output, 1)
        correct += pred.eq(target).sum().item()
        miss += target.shape[0] - pred.eq(target).sum().item()
        loss = criterion(output, target)
        losses += loss.item()
        if (batch_idx + 1) % 100 == 0:
            t1 = time.time()
            t_batch = t1 - t0
            t0 = time.time()
            print('Test ------ Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_batch))
    return losses / len(data_loader), correct / (correct + miss)


def draw_curve(dir, x_epoch, train_loss, train_prec, test_loss, test_prec):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label='train')
    ax0.plot(x_epoch, test_loss, 'ro-', label='test')
    ax1.plot(x_epoch, train_prec, 'bo-', label='train')
    ax1.plot(x_epoch, test_prec, 'ro-', label='test')
    ax0.legend()
    ax1.legend()
    fig.savefig(osp.join(dir, 'train.jpg'))
