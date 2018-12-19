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


def draw_curve(filename, x_epoch, train_loss, train_prec, test_loss, test_prec):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="prec")
    ax0.plot(x_epoch, train_loss, 'bo-', label='train')
    ax0.plot(x_epoch, test_loss, 'ro-', label='test')
    ax1.plot(x_epoch, train_prec, 'bo-', label='train')
    ax1.plot(x_epoch, test_prec, 'ro-', label='test')
    ax0.legend()
    ax1.legend()
    fig.savefig(filename)


def pairwise_distances(x, y=None):
    # Input: x is a Nxd matrix
    #        y is an optional Mxd matirx
    # Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
    #         if y is not given then use 'y=x'.
    # i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf).sqrt()
