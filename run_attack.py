import argparse
import json
import logging
import sys
from glob import glob
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from tqdm.auto import tqdm

from load_dataset.prep import load_data
from utils import set_seed
from paper_models import create_model
from LPSA.lpsa import LPSAttack

def setup_logging():
    logger = logging.getLogger()
    logger.handlers = [] 
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def my_get_local_attack_nodes(attr, adj, labels, surrogate_model, idx_test, device, n_nodes=500, min_node_degree=2):
    with torch.no_grad():
        surrogate_model = surrogate_model.to(device)
        surrogate_model.eval()

        attr = attr.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        
        logits = surrogate_model(attr, adj)[idx_test]
        preds = logits.argmax(dim=-1)

        correct_mask = (preds == labels[idx_test])
        acc = correct_mask.float().mean().item()
        
    logging.info(f"Starting random sampling of {n_nodes} nodes, current target model accuracy: {acc:.4f}")
    valid_nodes = idx_test[correct_mask]
    degrees = adj.sum(dim=1).to(device)
    degree_mask = (degrees[valid_nodes] >= min_node_degree)
    
    candidate_pool = valid_nodes[degree_mask]

    logging.info(
        f"Among {len(idx_test)} test nodes, found {len(candidate_pool)} "
        f"candidate nodes with degree >= {min_node_degree} and correct classification."
    )

    candidate_pool_np = candidate_pool.cpu().numpy()
    sample_size = min(n_nodes, len(candidate_pool_np))
    final_nodes = np.random.choice(candidate_pool_np, size=sample_size, replace=False)

    logging.info(f"Successfully randomly sampled {len(final_nodes)} nodes for attack.")
    
    return final_nodes.tolist()

def load_model_from_pt(model_backbone, n_features, n_classes, pt_path, device):
    # ── Load model ─────────────────────────────────────────────────────────────
    model_params = dict(model=model_backbone)
    hyperparams  = dict(**model_params, n_features=n_features, n_classes=n_classes,
                        ppr_cache_params=None)
    ## train arxiv or mag with 256 filters, and the rest with 64 filters
    if args.dataset == "arxiv" or args.dataset == "mag":
        hyperparams["hidn"] = 256
    else:           # for Citeseer and Pubmed
        hyperparams["hidn"] = 64
    if model_backbone == "GAT":
        hyperparams["hids"] = 100  
        hyperparams["heads"] = 4
    else:
        hyperparams["hids"] = 8
        hyperparams["heads"] = 8




    model = create_model(hyperparams).to(device)
    
    ckpt = torch.load(pt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--data_dir",           type=str,   default="/path/to/dataset", help="Root directory for datasets")
parser.add_argument("--dataset",            type=str,   default="mag", choices=["Citeseer", "Pubmed", "arxiv", "mag"])
parser.add_argument("--seed",               type=int,   default=5)

# Model checkpoint paths
parser.add_argument("--cpt_saved_dir",       type=str,   default="cache")
# Victim model backbone
parser.add_argument("--victim_model",       type=str,   default="SGC", choices=["GCN", "SGC", "GAT", "GraphSAGE",])

# Attack hyperparameters
parser.add_argument("--attack",             type=str,   default="LPSAttack")
parser.add_argument("--epsilons",           type=float, nargs='+', default=[0.1])
parser.add_argument("--attack_epochs",      type=int,   default=1)
parser.add_argument("--block_size",         type=int,   default=4000)
parser.add_argument("--attack_nodes_number",         type=int,   default=500)
parser.add_argument("--min_node_degree",    type=int,   default=None)

# Output
parser.add_argument("--output",             type=str,   default="output/attack")

def main(args):
    setup_logging()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # ── 2. Build paths and load models ────────────────────────────────
    cache_path = Path(args.cpt_saved_dir)
    victim_dir = cache_path / "victim"
    surrogate_dir = cache_path / "surrogate"
    
    victim_filename = f"{args.dataset}-{args.victim_model}-seed-{args.seed}.pt"
    
    surrogate_filename = f"{args.dataset}-GCN-seed-{args.seed}.pt"

    surrogate_pt = surrogate_dir.joinpath(surrogate_filename)
    victim_pt    = victim_dir.joinpath(victim_filename)

    logging.info("=" * 60)
    logging.info(f"Attack Started .... ")
    logging.info(f"Dataset       : {args.dataset}")
    logging.info(f"victim model : {args.victim_model} <- {victim_pt}")
    logging.info(f"surrogate model : GCN <- {surrogate_pt}")
    logging.info("=" * 60)

    # ── Load data  ─────────────────────────────────────────────────────────
    attr, adj, labels, split = load_data(args.dataset, device, dataset_root=args.data_dir)
    idx_test = torch.tensor(split["test"], dtype=torch.long, device=device)
    
    n_features = attr.shape[1]
    n_classes  = int(labels[~labels.isnan()].max() + 1)

    # ── Load victim model and surrogate model ───────────────────────────────────────────
    surrogate_model = load_model_from_pt('GCN', n_features, n_classes, surrogate_pt, device)
    victim_model    = load_model_from_pt(args.victim_model, n_features, n_classes, victim_pt, device)

    # ── Filter attack target nodes ─────────────────────────────────────────────────
    epsilons = sorted(args.epsilons)
    minimal_degree = args.min_node_degree or int(1 / min(epsilons))
    
    target_nodes = my_get_local_attack_nodes(
        attr, adj, labels, victim_model, idx_test, device, 
        n_nodes=args.attack_nodes_number, min_node_degree=minimal_degree
    )
    target_nodes = [int(i) for i in target_nodes]
    
    if not target_nodes:
        logging.error("No attack nodes found!")
        return

    # ── Initialize attacker ─────────────────────────────────────────────────────
    adversary = LPSAttack(
        attr=attr,
        adj=adj,
        labels=labels,
        model=surrogate_model,
        idx_attack=idx_test,
        device=device,
        data_device=device,
        make_undirected=True,
        epochs=args.attack_epochs, 
        block_size=args.block_size
    )
    adversary.set_eval_model(victim_model)

    results = []

    for node in tqdm(target_nodes, desc="Attacking nodes"):
        degree = adj[node].sum()
        for eps in epsilons:
            n_perturbations = int((eps * degree).round().item())
            if n_perturbations == 0:
                logging.warning(f"Skipping node {node} eps {eps}: 0 perturbations.")
                continue

            adversary.attack(n_perturbations, node_idx=node)
            logits_evasion, initial_logits_evasion = adversary.evaluate_local(node)

            entry = {
                "epsilon": eps,
                "n_perturbations": n_perturbations,
                "degree": int(degree.item()),
                "target": labels[node].item(),
                "node_id": node,
                "perturbed_edges": adversary.get_perturbed_edges().cpu().numpy().tolist() if hasattr(adversary, 'get_perturbed_edges') else [],
                "evasion": {
                    "logits": logits_evasion.cpu().numpy().tolist(),
                    "initial_logits": initial_logits_evasion.cpu().numpy().tolist(),
                    **adversary.classification_statistics(logits_evasion.cpu(), labels[node].long().cpu()),
                    **{f"initial_{k}": v for k, v in adversary.classification_statistics(
                        initial_logits_evasion.cpu(), labels[node].long().cpu()).items()},
                },
            }

            results.append(entry)
            # logging.info(entry)

    success_count = sum(1 for r in results if r["evasion"]["margin"] < 0)
    total_attacks = len(results)
    asr = (success_count / total_attacks) * 100 if total_attacks > 0 else 0
    logging.info("=" * 60)
    logging.info(f"  Attack Success Rate (ASR): {success_count}/{total_attacks} ({asr:.2f}%)")
    logging.info("=" * 60)

    # ── Save results ──────────────────────────────────
    assert len(results) > 0, "No results produced!"
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)   

    prefix = f"{args.attack}_{args.dataset}_eps_{'_'.join(str(e) for e in epsilons)}"
    existing = glob(str(output_dir / prefix) + "_*.json")
    uid = (max(int(Path(f).stem.rsplit("_", 1)[-1]) for f in existing) + 1) if existing else 0

    out_path = output_dir / f"{prefix}_{uid:06d}.json"
    out_path.write_text(json.dumps(results, indent=4))
    logging.info(f"Saved → {out_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
