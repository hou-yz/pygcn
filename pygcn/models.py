import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import *
from torch.nn import init


class GCN(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN, self).__init__()

        self.node_gc1 = NodeGCN(nfeat, 128)
        self.node_gc2 = NodeGCN(128, 128)
        # self.edge_gc1 = EdgeGCN(128, 1)
        self.edge_gc2 = EdgeGCN(128, nclass)

    def forward(self, feat, adj):
        x = self.node_gc1(feat, adj)
        x = F.relu(x)
        # a = self.edge_gc1(x, adj)
        # a = F.relu(a).view(x.shape[0], x.shape[0])
        x = self.node_gc2(x, adj)
        x = F.relu(x)
        a = self.edge_gc2(x, adj)
        return a


class Metric(nn.Module):
    def __init__(self, nfeat, nclass):
        super(Metric, self).__init__()
        self.num_class = nclass
        self.fc1 = nn.Linear(nfeat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, self.num_class)
        init.normal_(self.out_layer.weight, std=0.001)
        init.constant_(self.out_layer.bias, 0)

    def forward(self, feat, adj):
        feat_diff = feat.unsqueeze(0).repeat(feat.shape[0], 1, 1) - feat.unsqueeze(1).repeat(1, feat.shape[0], 1)
        feat_diff = feat_diff.view(feat.shape[0] * feat.shape[0], -1)
        out = self.fc1(feat_diff)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.out_layer(out)
        return out
