import numpy as np
import torch
from torch_sparse import coalesce
from typing import Union, Sequence, Tuple
import random


try:
    import resource
    _resource_module_available = True
except ModuleNotFoundError:
    _resource_module_available = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 1. 评估与监控工具
# ==========================================
def accuracy(logits: torch.Tensor, labels: torch.Tensor, split_idx: np.ndarray) -> float:
    """计算分类准确率"""
    return (logits.argmax(1)[split_idx] == labels[split_idx]).float().mean().item()


def get_max_memory_bytes():
    """获取当前程序使用的最大物理内存 (RAM) - """
    if _resource_module_available:
        return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return np.nan

# ==========================================
# 2. 攻击算法所需的核心张量工具
# ==========================================
def grad_with_checkpoint(outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    """带有 checkpoint 的梯度计算（LPSA 提取梯度必用）"""
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)

    for input_tensor in inputs:
        if not input_tensor.is_leaf:
            input_tensor.retain_grad()

    torch.autograd.backward(outputs)

    grad_outputs = []
    for input_tensor in inputs:
        grad_outputs.append(input_tensor.grad.clone())
        input_tensor.grad.zero_()
    return tuple(grad_outputs)


def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                 n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    """将有向边转为无向对称边（攻击后重构无向图必用）"""
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight
