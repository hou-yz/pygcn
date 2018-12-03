import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NodeGCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(NodeGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        support = torch.matmul(feat, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EdgeGCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(EdgeGCN, self).__init__()
        self.in_features = in_features + 1
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        feat_diff = feat.unsqueeze(0).repeat(feat.shape[0], 1, 1) - feat.unsqueeze(1).repeat(1, feat.shape[0], 1)
        feat_n_adj = torch.cat((feat_diff, adj.unsqueeze(2)), 2).view(feat.shape[0] * feat.shape[0], -1)
        # feat_diff.view(feat.shape[0] * feat.shape[0], -1)
        output = torch.matmul(feat_n_adj, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
