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
from glob import glob


class GCN_L1(Dataset):
    def __init__(self, root):
        self.root = root
        self.filenames = []
        for file in sorted(glob(osp.join(self.root, '*.mat'))):
            self.filenames.append(osp.basename(file))

        pass

    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        with h5py.File(osp.join(self.root, self.filenames[index]), 'r') as f:
            edges_target = np.array(f['edge_target']).transpose()
            node_feat = np.array(f['node_feat']).transpose()
            edge_iou = np.array(f['edge_iou']).transpose()
        return edges_target, node_feat, edge_iou
