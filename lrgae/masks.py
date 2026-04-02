import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
    
from copy import copy
from typing import Optional, Tuple

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree, sort_edge_index, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes


def mask_path(edge_index: Tensor, 
              p: float = 0.3,               # 70%的节点/边作为路径起点
              walks_per_node: int = 1,      # 每个起点游走1次  
              walk_length: int = 3,         # 游走长度=2（包含3个节点）
              num_nodes: Optional[int] = None,
              start: str = 'node',          # 从节点开始游走
              is_sorted: bool = False,
              training: bool = True
              ) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:            
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    assert start in ['node', 'edge']
    num_edges = edge_index.size(1)          # 边数 9874 训练图的边数
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    
    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    # 从边索引中安全地推断节点数量，避免手动指定
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    if not is_sorted:
        edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
        start = row[sample_mask].repeat(walks_per_node)
    else:
        # 随机选择 70% 的节点作为起点
        start = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes*p)].repeat(walks_per_node)         # num_nodes*p = 1896
    # 计算每个节点的度数
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)       # rowptr 是 CSR（Compressed Sparse Row）用于高效存储稀疏矩阵, rowptr[i] 表示：节点i的边在边列表中的起始位置
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)         # [1896, 2]         70%的节点随即游走，走出来的边id：e_id
    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges   e_id.shape = 3625     # 边的索引  非法边 指的是在随机游走过程中遇到的 -1 值  当随机游走到孤立节点或出度为0的节点时，无法继续游走，返回 -1
    edge_mask[e_id] = False                 # 随即挑中的这70%的节点进行随即游走得到的边的集合，保留；其余剩下的边是要被mask的
    # edge_index[:, edge_mask] → 选择 edge_mask 为 True 的列    # edge_index [2, 8974]   [2, 5865]这是要保留的       [2, 3109]这是mask的
    # 外面remaining_edges, masked_edges 接收            后面这个索引出来的是70%的顶点随即游走走出来的边的id
    return edge_index[:, edge_mask], edge_index[:, ~edge_mask]


def mask_edge(edge_index: Tensor, p: float=0.7):
    if p < 0. or p > 1.:
        raise ValueError(f'Mask probability has to be between 0 and 1 '
                         f'(got {p}')    
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]

def mask_feature(
    x: Tensor,
    p: float = 0.5,
    mode: str = 'col',
    fill_value: float = 0.,
    training: bool = True,
) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Masking ratio has to be between 0 and 1 '
                         f'(got {p}')
    if not training or p == 0.0:
        return x, torch.ones_like(x, dtype=torch.bool)
    assert mode in ['row', 'col', 'all']

    if mode == 'row':
        mask = torch.rand(x.size(0), device=x.device) >= p
        mask = mask.view(-1, 1)
    elif mode == 'col':
        mask = torch.rand(x.size(1), device=x.device) >= p
        mask = mask.view(1, -1)
    else:
        mask = torch.rand_like(x) >= p

    remaining_features = x.masked_fill(~mask, fill_value)
    return remaining_features, mask

class MaskPath(nn.Module):
    def __init__(self, p: float = 0.7, 
                 walks_per_node: int = 1,
                 walk_length: int = 3, 
                 start: str = 'node',
                 num_nodes: Optional[int]=None,
                 undirected: bool=True):
        super().__init__()
        self.p = p
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.start = start
        self.num_nodes = num_nodes
        self.undirected = undirected

    def forward(self, data):
        edge_index = data.edge_index        # 边的索引  # [2, num_edges]
        # [2, 5865]这是要保留的       [2, 3109]这是mask的
        remaining_edges, masked_edges = mask_path(edge_index, self.p,
                                                  walks_per_node=self.walks_per_node,
                                                  walk_length=self.walk_length,
                                                  start=self.start,
                                                  num_nodes=self.num_nodes)
        remaining_graph = copy(data)        
        masked_graph = copy(data)
        remaining_graph.masked_edges = masked_edges             # 这里需要指出被mask掉的边的索引. 这里的边是上面的3109
        masked_graph.masked_edges = remaining_edges
        # 5865 + 3109 = 8974
        if self.undirected:     # 将有向边转换为无向边，自动添加反向边; 自动处理重复边
            remaining_edges = to_undirected(remaining_edges)    # 5865 --> 8186   
            masked_edges = to_undirected(masked_edges)          # 3109 --> 4928      
            
        masked_graph.edge_index = masked_edges
        remaining_graph.edge_index = remaining_edges
        return remaining_graph, masked_graph

    def extra_repr(self):
        return f"p={self.p}, walks_per_node={self.walks_per_node}, walk_length={self.walk_length}, \n"\
            f"start={self.start}, undirected={self.undirected}"


class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7, undirected: bool=True):
        super().__init__()
        self.p = p
        self.undirected = undirected

    def forward(self, data):
        edge_index = data.edge_index
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)

            
        remaining_graph = copy(data)
        masked_graph = copy(data)
        remaining_graph.masked_edges = masked_edges
        masked_graph.masked_edges = remaining_edges
        
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)        
            masked_edges = to_undirected(masked_edges)        
            
        masked_graph.edge_index = masked_edges
        remaining_graph.edge_index = remaining_edges
        return remaining_graph, masked_graph

    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"

class MaskHeteroEdge(nn.Module):
    def __init__(self, p: float=0.7):
        super().__init__()
        self.p = p

    def forward(self, data):
        remaining_graph = copy(data)
        masked_graph = copy(data)        
        
        for edge_type in data.edge_types:
            src, _, dst = edge_type
            edge_index = data[edge_type].edge_index
            remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
            remaining_graph[edge_type].edge_index = remaining_edges
            remaining_graph[edge_type].masked_edges = masked_edges
            
            masked_graph[edge_type].edge_index = masked_edges
            masked_graph[edge_type].masked_edges = remaining_edges
        
        return remaining_graph, masked_graph

    def extra_repr(self):
        return f"p={self.p}"   
        
class MaskFeature(nn.Module):
    def __init__(self, p: float=0.7):
        super().__init__()
        self.p = p
        
    def forward(self, data):
        edge_index = data.edge_index
        # `mask` is the mask indicating the reserved nodes
        remaining_features, mask = mask_feature(data.x, p=self.p, mode='row')
        masked_features = data.x.masked_fill(mask, 0.0)
        # assert torch.allclose(data.x, remaining_features+masked_features)
        
        remaining_graph = copy(data)
        remaining_graph.x = remaining_features
        remaining_graph.masked_nodes = ~mask
        masked_graph = copy(data)
        masked_graph.x = masked_features
        masked_graph.masked_nodes = mask
        return remaining_graph, masked_graph     
        
    def extra_repr(self):
        return f"p={self.p}"        
        
class NullMask(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, data):
        remaining_graph = copy(data)
        if isinstance(data, Data):
            remaining_graph.masked_nodes = data.x.new_ones(data.x.size(0), 1, dtype=torch.bool)
            remaining_graph.masked_edges = data.edge_index
        return remaining_graph, copy(data)


class AdversMask(nn.Module):
    def __init__(self, generator, fc_in_channels, fc_out_channels):
        super().__init__()
        self.generator = generator
        self.fc = nn.Linear(fc_in_channels, fc_out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.generator(x, edge_index)[-1]
        z = F.gumbel_softmax(self.fc(x), hard=True)
        return z
