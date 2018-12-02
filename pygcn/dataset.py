from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path as osp
import errno
import numpy as np
import torch
import codecs
import h5py
from collections import defaultdict
from torch.utils.data import Dataset


class GCN_L1(Dataset):
    def __init__(self, root):
        self.root = root
        self.edge_targets = defaultdict()
        self.node_feats = defaultdict()
        self.edge_ious = defaultdict()
        with h5py.File(osp.join(self.root, 'edge_targets.mat'), 'r') as f:
            for i in range(len(f['edge_targets'])):
                self.edge_targets[i] = np.array(f[f['edge_targets'][i, 0]])
        with h5py.File(osp.join(self.root, 'node_feats.mat'), 'r') as f:
            for i in range(len(f['node_feats'])):
                self.node_feats[i] = np.array(f[f['node_feats'][i, 0]])
        with h5py.File(osp.join(self.root, 'edge_ious.mat'), 'r') as f:
            for i in range(len(f['edge_ious'])):
                self.edge_ious[i] = np.array(f[f['edge_ious'][i, 0]])

        pass


    def __getitem__(self, index):
        edges_target = self.edge_targets[index]
        node_feat = self.node_feats[index]
        edge_iou = int(self.edge_ious[index])
        return edges_target, node_feat, edge_iou
