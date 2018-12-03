from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import *
from pygcn.models import *
from pygcn.dataset import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=40,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

args = parser.parse_args()

np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
train_set = GCN_L1(osp.expanduser('~/Data/DukeMTMC/ground_truth/GCN_L1/train'))
test_set = GCN_L1(osp.expanduser('~/Data/DukeMTMC/ground_truth/GCN_L1/val'))
train_loader = DataLoader(train_set, batch_size=args.batch_size,
                          num_workers=4, pin_memory=True, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size,
                         num_workers=4, pin_memory=True)
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=256, nclass=2)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

model.cuda()
criterion.cuda()

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch, model, test_loader, optimizer, criterion, )
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test(model, test_loader, criterion, )
