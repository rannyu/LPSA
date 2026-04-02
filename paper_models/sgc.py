import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def torch_sparse_spmm(edge_index, edge_weight, x):
    n = x.size(0)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                       value=edge_weight, sparse_sizes=(n, n))
    return adj @ x

class SGC(nn.Module):
    def __init__(self, n_features: int, n_classes: int,
                 K: int = 2, bias: bool = True, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.K        = K
        self.dropout  = nn.Dropout(p=dropout)
        self.lin      = nn.Linear(n_features, n_classes, bias=bias)

    def forward(self, data: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        x = data
        row, col, val = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        
        edge_index, edge_weight = gcn_norm(
            edge_index, val, x.size(0),
            add_self_loops=True, dtype=x.dtype
        )
        
        for _ in range(self.K):
            x = torch_sparse_spmm(edge_index, edge_weight, x)

        return self.lin(self.dropout(x))