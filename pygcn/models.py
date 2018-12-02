import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import NodeGCN


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = NodeGCN(nfeat, 128)
        self.gc2 = NodeGCN(128, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
