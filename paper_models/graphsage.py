import torch
import torch.nn.functional as F
from tqdm import tqdm


from torch_geometric.nn import SAGEConv

class SAGE(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidn=256, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(n_features, hidn))
        self.convs.append(SAGEConv(hidn, n_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(x_all.device)
                x = conv(x, batch.edge_index.to(x_all.device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all