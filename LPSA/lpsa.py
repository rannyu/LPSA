import math
import logging
from collections import defaultdict
from typing import Union, Sequence, Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F
import torch_sparse
from torch_sparse import SparseTensor, coalesce

# =========================================================================
# 核心数学算子
# =========================================================================
def grad_with_checkpoint(outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
                         inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for t in inputs:
        if not t.is_leaf: t.retain_grad()
    torch.autograd.backward(outputs)
    grads = tuple(t.grad.clone() for t in inputs)
    for t in inputs: t.grad.zero_()
    return grads

def to_symmetric(edge_index: torch.Tensor, edge_weight: torch.Tensor, n: int, op='mean') -> Tuple[torch.Tensor, torch.Tensor]:
    sym_idx = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
    sym_wt = edge_weight.repeat(2)
    return coalesce(sym_idx, sym_wt, m=n, n=n, op=op)

# =========================================================================
# 极致单步攻击类 LPSAttack (零继承，完美独立)
# =========================================================================
class LPSAttack:
    def __init__(self, adj, attr, labels, model, device, data_device,
                 make_undirected=True, lr_factor=0.05,
                 display_step=20, epochs=1, block_size=4000, eps=1e-14, 
                 with_early_stopping=True, do_synchronize=False, **kwargs):
        
        self.device = device
        self.data_device = data_device
        self.make_undirected = make_undirected
        self.eps = eps
        self.block_size = block_size
        self.epochs = epochs
        self.display_step = display_step
        self.with_early_stopping = with_early_stopping
        self.do_synchronize = do_synchronize
        
        # 初始化模型 (冻结梯度)
        self.attacked_model = deepcopy(model).to(self.device).eval()
        for p in self.attacked_model.parameters(): p.requires_grad = False
        self.eval_model = self.attacked_model

        # 数据挂载
        self.attr = attr.to(self.data_device)
        self.adj = adj.to(self.data_device)
        self.n = self.adj.size(0)
        self.labels = labels.to(torch.long).to(self.device)

        # 超参与状态
        self.lr_factor = lr_factor * max(math.sqrt(self.n / self.block_size), 1.)
        self.n_possible_edges = self.n * (self.n - 1) // 2
        self.degrees = self.adj.sum(dim=1).to(self.device)

        self.current_search_space = None
        self.modified_edge_weight_diff = None
        self.adj_adversary = self.adj
        self.perturbed_edges = torch.tensor([])

    # -------------------------------------------------------------------------
    # 基础方法与工具
    # -------------------------------------------------------------------------
    def set_eval_model(self, model):
        self.eval_model = deepcopy(model).to(self.device).eval()

    def get_logits(self, model, node_idx, perturbed_graph=None):
        graph = perturbed_graph if perturbed_graph is not None else self.adj
        return model(self.attr.to(self.device), graph.to(self.device))[node_idx:node_idx + 1]

    def get_surrogate_logits(self, node_idx: int, perturbed_graph=None) -> torch.Tensor:
        return self.get_logits(self.attacked_model, node_idx, perturbed_graph)

    def get_eval_logits(self, node_idx: int, perturbed_graph=None) -> torch.Tensor:
        return self.get_logits(self.eval_model, node_idx, perturbed_graph)

    @torch.no_grad()
    def evaluate_local(self, node_idx: int):
        self.eval_model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            logging.info(f"Cuda Memory before local evaluation on clean adjacency {torch.cuda.memory_allocated() / (1024 ** 3)}")

        initial_logits = self.get_eval_logits(node_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            logging.info(f"Cuda Memory before local evaluation on perturbed adjacency {torch.cuda.memory_allocated() / (1024 ** 3)}")

        logits = self.get_eval_logits(node_idx, self.adj_adversary)
        return logits, initial_logits

    def classification_statistics(self, logits, label):
        logits, label = F.log_softmax(logits.cpu(), dim=-1), label.cpu()
        logits = logits[0]
        logit_target = logits[label].item()
        sorted = logits.argsort()
        logit_best_non_target = (logits[sorted[sorted != label][-1]]).item()
        confidence_target = np.exp(logit_target)
        confidence_non_target = np.exp(logit_best_non_target)
        margin = confidence_target - confidence_non_target
        return {
            'logit_target': logit_target,
            'logit_best_non_target': logit_best_non_target,
            'confidence_target': confidence_target,
            'confidence_non_target': confidence_non_target,
            'margin': margin
        }

    def calculate_loss(self, logits, labels):
        
        scaled = logits / 1.0
        target_logit = scaled[np.arange(logits.size(0)), labels.squeeze()]
        sorted_idx = scaled.argsort(-1)
        best_non_tgt = sorted_idx[sorted_idx != labels[:, None]].reshape(logits.size(0), -1)[:, -1]
        return (scaled[np.arange(logits.size(0)), best_non_tgt] - target_logit).mean()

    def project(self, n_perturbations: int, values: torch.Tensor, eps: float = 0, inplace: bool = False):
        if not inplace:
            values = values.clone()

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = self.bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def bisection(self, edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
        def func(x):
            return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

        miu = a
        for i in range(int(iter_max)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu) * func(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= epsilon):
                break
        return miu

    def _append_attack_statistics(self, loss, statistics):
        self.attack_statistics['loss'].append(loss)
        self.attack_statistics['perturbation_mass'].append(torch.clamp(self.modified_edge_weight_diff, 0, 1).sum().item())
        for k, v in statistics.items(): self.attack_statistics[k].append(v)

    # -------------------------------------------------------------------------
    # 图结构操作
    # -------------------------------------------------------------------------
    def perturb_graph(self, node_idx):
        mod_adj = SparseTensor(row=torch.zeros_like(self.current_search_space),
                               col=self.current_search_space,
                               value=self.modified_edge_weight_diff,
                               sparse_sizes=(1, self.n))
        v_rows, v_cols, v_vals = mod_adj.coo()
        v_rows += node_idx
        v_idx = torch.stack([v_rows, v_cols], dim=0)

        A_rows, A_cols, A_vals = self.adj.coo()
        A_idx = torch.stack([A_rows, A_cols], dim=0)

        is_row = A_rows == node_idx
        A_idx_row, A_vals_row = A_idx[:, is_row], A_vals[is_row]

        A_idx_row = torch.cat((v_idx, A_idx_row), dim=-1)
        A_vals_row = torch.cat((v_vals, A_vals_row))
        A_idx_row, A_vals_row = torch_sparse.coalesce(A_idx_row, A_vals_row, m=self.n, n=self.n, op='sum')

        is_before, is_after = A_rows < node_idx, A_rows > node_idx
        A_idx = torch.cat((A_idx[:, is_before], A_idx_row, A_idx[:, is_after]), dim=-1)
        A_weights = torch.cat((A_vals[is_before], A_vals_row, A_vals[is_after]), dim=-1)
        
        A_weights[A_weights > 1] = 2 - A_weights[A_weights > 1]
        if self.make_undirected:
            A_idx, A_weights = to_symmetric(A_idx, A_weights, self.n, op='max')

        return SparseTensor.from_edge_index(A_idx, A_weights, (self.n, self.n))

    def get_leaf_specialists(self, node_idx, target_label):
        source_label = self.labels[node_idx].item()
        degree_one_nodes = (self.degrees == 1).nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            logits = self.attacked_model(self.attr.to(self.device), self.adj.to(self.device))
        
        max_probs, pred_labels = torch.max(torch.softmax(logits[degree_one_nodes], dim=-1), dim=-1)
        quality_mask = (max_probs > 0.5) & (pred_labels != source_label)
        
        leaf_nodes = degree_one_nodes[quality_mask][torch.argsort(max_probs[quality_mask], descending=True)]
        print(f'排除源类别 {source_label} 后，高质量叶子共: {len(leaf_nodes)} 个')
        return leaf_nodes[: int(self.block_size * 1)]

    def sample_search_space(self, node_idx: int, n_perturbations: int, target_label: int):
        leaf_specialists = self.get_leaf_specialists(node_idx, target_label)
        self.current_search_space = leaf_specialists.unique()
        self.current_search_space = self.current_search_space[self.current_search_space != node_idx]

        if self.current_search_space.size(0) < self.block_size:
            needed = self.block_size - self.current_search_space.size(0)
            idx_rand = torch.randint(self.n - 1, (needed,), device=self.device)
            idx_rand[idx_rand >= node_idx] += 1
            self.current_search_space = torch.cat([self.current_search_space, idx_rand]).unique()

        self.modified_edge_weight_diff = torch.full_like(
            self.current_search_space, self.eps, dtype=torch.float32, requires_grad=True
        )

    def update_edge_weights(self, n_perturbations: int, epoch: int, gradient: torch.Tensor):
        lr = n_perturbations * self.lr_factor
        self.modified_edge_weight_diff.data.add_(lr * gradient)

    @torch.no_grad()
    def sample_final_edges(self, node_idx: int, n_perturbations: int) -> SparseTensor:
        # 1. 提取权重绝对值，过滤掉未被优化的初始极小值 eps
        s = self.modified_edge_weight_diff.abs().detach()
        s[s == self.eps] = 0
        
        # 2. 直接获取权重最大的前 K 个位置的索引
        top_indices = torch.topk(s, n_perturbations).indices
        
        # 3. 直接用索引保留候选边，并把它们的权重一口气设为 1.0
        self.current_search_space = self.current_search_space[top_indices]
        self.modified_edge_weight_diff = torch.ones_like(self.current_search_space, dtype=torch.float32)
        
        # 4. 组装并返回最终的对抗图
        return self.perturb_graph(node_idx)

    def calc_perturbed_edges(self, node_idx: int) -> torch.Tensor:
        source = torch.full_like(self.current_search_space, node_idx).cpu()
        target = self.current_search_space.cpu()
        return torch.stack((source, target), dim=0)

    def get_perturbed_edges(self) -> torch.Tensor:
        if not hasattr(self, "perturbed_edges"): return torch.tensor([])
        return self.perturbed_edges

    # -------------------------------------------------------------------------
    # 单步攻击执行逻辑
    # -------------------------------------------------------------------------
    def attack(self, n_perturbations: int, node_idx: int, **kwargs):
        if n_perturbations <= 0:
            self.adj_adversary = self.adj
            return

        # 动态确定目标类别
        with torch.no_grad():
            logits_clean = self.get_surrogate_logits(node_idx).to(self.device)
            src_y_idx = self.labels[node_idx].item()
            sorted_logits = logits_clean.argsort(-1)
            target_label = sorted_logits[0, -1].item() if sorted_logits[0, -1] != src_y_idx else sorted_logits[0, -2].item()
            del logits_clean

        with torch.no_grad():
            self.sample_search_space(node_idx, n_perturbations, target_label)
            best_margin, best_epoch = float('Inf'), float('-Inf')
            self.attack_statistics = defaultdict(list)
        
            logits_orig = self.get_surrogate_logits(node_idx).to(self.device)
            loss_orig = self.calculate_loss(logits_orig, self.labels[node_idx, None]).to(self.device)
            statistics_orig = self.classification_statistics(logits_orig, self.labels[node_idx])
            logging.info(f'Original: Loss: {loss_orig.item()} Statstics: {statistics_orig}\n')
            del logits_orig, loss_orig
  
        self.modified_edge_weight_diff.requires_grad = True
        perturbed_graph = self.perturb_graph(node_idx)

        if torch.cuda.is_available() and self.do_synchronize:
            torch.cuda.empty_cache(); torch.cuda.synchronize()

        logits = self.get_surrogate_logits(node_idx, perturbed_graph)
        loss = self.calculate_loss(logits, self.labels[node_idx][None])
        
        classification_statistics = self.classification_statistics(logits, self.labels[node_idx].to(self.device))
        logging.info(f'Initial: Loss: {loss.item()} Statstics: {classification_statistics}\n')

        gradient = grad_with_checkpoint(loss, self.modified_edge_weight_diff)[0]

        epoch = 0
        if torch.cuda.is_available() and self.do_synchronize:
            torch.cuda.empty_cache(); torch.cuda.synchronize()

        with torch.no_grad():
            self.modified_edge_weight_diff.requires_grad = False
            self.update_edge_weights(n_perturbations, epoch, gradient)
            self.modified_edge_weight_diff = self.project(n_perturbations, self.modified_edge_weight_diff, self.eps)

            perturbed_graph = self.perturb_graph(node_idx)
            logits = self.get_surrogate_logits(node_idx, perturbed_graph).to(self.device)
            classification_statistics = self.classification_statistics(logits, self.labels[node_idx].to(self.device))
            
            if epoch % self.display_step == 0:
                logging.info(f'\nEpoch: {epoch} Loss: {loss.item()} Statstics: {classification_statistics}\n')
                logging.info(f"Gradient mean {gradient.abs().mean().item()} std {gradient.abs().std().item()} with base learning rate {n_perturbations * self.lr_factor}")
                if torch.cuda.is_available(): logging.info(f'Cuda memory {torch.cuda.memory_allocated() / (1024 ** 3)}')

            if self.with_early_stopping and best_margin > classification_statistics['margin']:
                best_margin, best_epoch = classification_statistics['margin'], epoch

            self._append_attack_statistics(loss.item(), classification_statistics)

        if best_margin > statistics_orig['margin']:
            self.perturbed_edges = torch.tensor([])
            self.adj_adversary = None
            logging.info(f"Failed to attack node {node_idx} with n_perturbations={n_perturbations}")
            return None



        self.adj_adversary = self.sample_final_edges(node_idx, n_perturbations)
        self.perturbed_edges = self.calc_perturbed_edges(node_idx)