import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import *
from gcn.utils import *
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
        # adj = torch.matmul(adj, adj)
        identity = torch.eye(adj.shape[0]).cuda()
        similarity = (19 - pairwise_distances(feat, feat)) / 11
        similarity = (similarity + 1) / 2
        # similarity = F.softmax(similarity, dim=1)
        x = self.node_gc1(feat, similarity)
        x = F.relu(x)
        x = self.node_gc2(x, similarity)
        x = F.relu(x)
        x = self.node_gc3(x, similarity)
        x = F.relu(x)
        a = self.edge_gc(x, adj)
        return a


class GAT(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(nfeat, 128)
        self.gat2 = GATLayer(128, 128)
        self.gat3 = GATLayer(128, 128)
        # self.edge_gc1 = EdgeGCN(128, 1)
        self.edge_gc = EdgeGCN(128, nclass, include_adj=False)

    def forward(self, feat, adj):
        # identity = torch.eye(adj.shape[0]).cuda()
        x = self.gat1(feat, adj)
        x = self.gat2(x, adj)
        x = self.gat3(x, adj)
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
        feat_diff = (
                feat.unsqueeze(0).repeat(feat.shape[0], 1, 1) - feat.unsqueeze(1).repeat(1, feat.shape[0], 1)).abs()
        feat_diff = feat_diff.view(feat.shape[0] * feat.shape[0], -1)
        out = self.fc1(feat_diff)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.out_layer(out)
        return out
