from typing import Any, Dict, Union

from paper_models.gcn import GCN
from paper_models.sgc import SGC
from paper_models.gat import GAT
from paper_models.graphsage import SAGE


MODEL_TYPE = Union[SGC, GAT, SAGE]

def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Parameters
    ----------
    hyperparams : Dict[str, Any]
        Containing the hyperparameters.

    Returns
    -------
    model: MODEL_TYPE
        The created instance.
    """
    if 'model' not in hyperparams or hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == "SGC":
        return SGC(**hyperparams)
    if hyperparams['model'] == "GAT":
        return GAT(**hyperparams)
    if hyperparams['model'] == "GraphSAGE":
        return SAGE(**hyperparams)
    return GCN(**hyperparams)




__all__ = [GCN,
           GAT, 
           create_model,
           SGC,]
