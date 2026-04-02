import torch
import torch_sparse
import numpy as np
import scipy.sparse as sp
import logging
from typing import Tuple, Union, Optional, Dict
from torchtyping import TensorType
from torch_sparse import SparseTensor
import torch_geometric.transforms as T

from load_dataset.loader import load_dataset

try:
    import resource
    def get_max_memory_bytes():
        return 1024 * resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
except ModuleNotFoundError:
    def get_max_memory_bytes():
        return float('nan')



def load_data(name: str,
              device: Union[int, str, torch.device] = 0,
              dataset_root: str = 'data') -> Tuple[TensorType["num_nodes", "num_features"],
                                                    SparseTensor,
                                                    TensorType["num_nodes"],
                                                    Optional[Dict[str, np.ndarray]]]:

    logging.debug(f"Memory before loading: {get_max_memory_bytes() / (1024**3):.2f} GB")

    transform = T.ToUndirected()
    data = load_dataset(dataset_root, name, transform=transform)
    logging.info(data)

    attr = torch.from_numpy(data.x.cpu().numpy()).to(device)

    edge_index  = data.edge_index.cpu()
    edge_weight = torch.ones(edge_index.size(1)) if data.edge_attr is None else data.edge_attr.cpu()

    adj = sp.csr_matrix((edge_weight, edge_index), (data.num_nodes, data.num_nodes))
    del edge_index, edge_weight

    adj = torch_sparse.SparseTensor.from_scipy(adj).coalesce().to(device)

    split = {k: v.numpy() for k, v in dict(
        train=data.train_mask.nonzero().squeeze(),
        valid=data.val_mask.nonzero().squeeze(),
        test=data.test_mask.nonzero().squeeze(),
    ).items()}

    labels = data.y.squeeze().to(device)

    return attr, adj, labels, split
