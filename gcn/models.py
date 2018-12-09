import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import *
from torch.nn import init


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, use_adj=True):
        super(GCN, self).__init__()
        self.use_adj = use_adj
        self.node_gc1 = NodeGCN(nfeat, 128)
        self.node_gc2 = NodeGCN(128, 128)
        self.node_gc3 = NodeGCN(128, 128)
        # self.edge_gc1 = EdgeGCN(128, 1)
        self.edge_gc = EdgeGCN(128, nclass, include_adj=False)

    def forward(self, feat, adj):
        identity = torch.eye(adj.shape[0]).cuda()
        x = self.node_gc1(feat, identity)
        x = F.relu(x)
        x = self.node_gc2(x, identity)
        x = F.relu(x)
        x = self.node_gc3(x, identity)
        x = F.relu(x)
        a = self.edge_gc(x, adj)
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
        feat_diff = (feat.unsqueeze(0).repeat(feat.shape[0], 1, 1) - feat.unsqueeze(1).repeat(1, feat.shape[0], 1)).abs()
        feat_diff = feat_diff.view(feat.shape[0] * feat.shape[0], -1)
        out = self.fc1(feat_diff)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.out_layer(out)
        return out
