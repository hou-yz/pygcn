import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import *


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
