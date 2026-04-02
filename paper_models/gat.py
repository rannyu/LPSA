import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv



class GAT(torch.nn.Module):
    def __init__(self, n_features, n_classes, hids, heads, **kwargs):
        super().__init__()
        self.conv1 = GATConv(n_features, hids, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hids * heads, n_classes, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, data, adj):
        curr_size = adj.size(1) 
        if data.size(0) > curr_size:
            data = data[:curr_size]

        x = F.dropout(data, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, adj))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, adj)
        return x