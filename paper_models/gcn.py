import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidn=16, **kwargs):
        # 注意：为了适配框架，参数名改为 n_features (in) 和 n_classes (out)
        # hidn 对应你代码里的 hidden_channels
        super().__init__()
        self.conv1 = GCNConv(n_features, hidn)
        # self.bn1 = torch.nn.BatchNorm1d(hidn)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)
        self.conv2 = GCNConv(hidn, n_classes)
        

    def forward(self, data, adj, **kwargs):
        x, edge_index = data, adj
        
        # Layer 0 (隐藏层)
        x = self.conv1(x, edge_index)
        x =  self.activation(x)
        x = self.dropout(x)
        
        # Layer 1 (输出层) - 无激活、无正则化
        x = self.conv2(x, edge_index)
        
        return x