import argparse
import json
import logging
from pyexpat import model
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Import our custom packages
from paper_models import create_model
from load_dataset.prep import load_data
from utils import accuracy, get_max_memory_bytes, set_seed


def setup_logging():
    """Set up standard logging"""
    logger = logging.getLogger()
    logger.handlers = [] # Clear existing handlers
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data parameters
parser.add_argument("--data-dir",           type=str,   default="/path/to/dataset.")
parser.add_argument("--dataset",            type=str,   default="arxiv", choices=["Citeseer", "Pubmed", "arxiv", "mag"])
parser.add_argument("--make-undirected",    action="store_true", default=True)
parser.add_argument("--binary-attr",        action="store_true", default=False)
parser.add_argument("--seed",               type=int,   default=5)

# Model parameters
parser.add_argument("--model",              type=str,   default="GAT", choices=["GCN", "SGC", "GAT", "GraphSAGE", ])
parser.add_argument("--model_type",         type=str,   default="victim", choices=["victim", "surrogate"], help="Model role: victim or surrogate")
# parser.add_argument("--n_filters",          type=int,   default=256, help="Number of hidden filters in the GNN layers")

# Training parameters
parser.add_argument("--lr",                 type=float, default=1e-2)
parser.add_argument("--weight-decay",       type=float, default=1e-7)
parser.add_argument("--patience",           type=int,   default=300)
parser.add_argument("--max-epochs",         type=int,   default=3000)
parser.add_argument("--display-steps",      type=int,   default=100)

# Fine-tuning parameters (Set to 0 to skip)
parser.add_argument("--fine-tune-epochs",   type=int,   default=40, help="If > 0, do Manifold-Smoothed Fine-tuning")
parser.add_argument("--lambda_dcg",         type=float, default=1)
parser.add_argument("--sigma",              type=float, default=0.03)
parser.add_argument("--temperature",        type=float, default=6.0)

# Output parameters
parser.add_argument("--artifact-dir",       type=str,   default="cache")
parser.add_argument("--output",             type=str,   default="output/train")

def main(args):
    # =========================================================================
    # Print Experiment Info
    # =========================================================================

    logging.info("=" * 60)
    logging.info("Training Task Started!")
    logging.info(f"Dataset : {args.dataset}")
    logging.info(f"Model   : {args.model}")
    logging.info(f"Role    : {args.model_type}")
    logging.info(f"Fine-tuning : {args.fine_tune_epochs}")
    logging.info("=" * 60)
    
    logging.info(vars(args))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load Data ─────────────────────────────────────────────────────────────
    attr, adj, labels, split = load_data(args.dataset, device, dataset_root=args.data_dir)
    idx_train, idx_val, idx_test = split["train"], split["valid"], split["test"]

    n_features = attr.shape[1]
    n_classes  = int(labels[~labels.isnan()].max() + 1)
    logging.info(f"train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    
    logging.info(f"Memory: {get_max_memory_bytes() / (1024**3):.2f} GB")
    
    # ── Load model ─────────────────────────────────────────────────────────────
    train_params = dict(lr=args.lr, weight_decay=args.weight_decay,
                        patience=args.patience, max_epochs=args.max_epochs)
    model_params = dict(model=args.model)
    hyperparams  = dict(**model_params, n_features=n_features, n_classes=n_classes,
                        ppr_cache_params=None, train_params=train_params)
    ## train arxiv or mag with 256 filters, and the rest with 64 filters
    if args.dataset == "arxiv" or args.dataset == "mag":
        hyperparams["hidn"] = 256
    else:           # for Citeseer and Pubmed
        hyperparams["hidn"] = 64
    if args.model == "GAT":
        hyperparams["hids"] = 100  
        hyperparams["heads"] = 4
    else:
        hyperparams["hids"] = 8
        hyperparams["heads"] = 8

    model = create_model(hyperparams).to(device)


    # ── Stage 1: Standard Pre-training ────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss, best_epoch, best_state = np.inf, 0, None

    model.train()
    for it in tqdm(range(args.max_epochs), desc="Pre-training"):
        optimizer.zero_grad()
        logits = model(attr, adj)
        loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss_val   = F.cross_entropy(logits[idx_val],   labels[idx_val])
        loss_train.backward()
        optimizer.step()

        # Save best model
        if loss_val < best_loss:
            best_loss  = loss_val
            best_epoch = it
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif it >= best_epoch + args.patience:
            break

        if it % args.display_steps == 0:
            logging.info(f"Epoch {it:4}: loss_train={loss_train.item():.5f}  "
                         f"loss_val={loss_val.item():.5f}  "
                         f"acc_train={accuracy(logits, labels, idx_train):.5f}  "
                         f"acc_val={accuracy(logits, labels, idx_val):.5f}")

    model.load_state_dict(best_state)

    # ── Stage 2: Manifold-Smoothed Fine-tuning ────────────────────────────────
    if args.fine_tune_epochs > 0:
        del logits, loss_train, loss_val
        torch.cuda.empty_cache()

        ft_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for it in tqdm(range(args.max_epochs), desc="Pre-training"):
            optimizer.zero_grad()
            logits     = model(attr, adj)
            loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
            loss_val   = F.cross_entropy(logits[idx_val],   labels[idx_val])
            loss_train.backward()
            optimizer.step()

            if loss_val < best_loss:
                best_loss  = loss_val.item()
                best_epoch = it
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            elif it >= best_epoch + args.patience:
                break

            if it % args.display_steps == 0:
                logging.info(f"Epoch {it:4}: loss_train={loss_train.item():.5f}  "
                            f"loss_val={loss_val.item():.5f}  "
                            f"acc_train={accuracy(logits, labels, idx_train):.5f}  "
                            f"acc_val={accuracy(logits, labels, idx_val):.5f}")

        model.load_state_dict(best_state)

        # ── Stage 2: Fine-tuning（可选）────────────────────────────────────
        if args.fine_tune_epochs > 0:
            del logits, loss_train, loss_val
            torch.cuda.empty_cache()

            ft_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            model.train()

            for ft_it in range(args.fine_tune_epochs):
                ft_optimizer.zero_grad()
                attr_noisy  = (attr + torch.randn_like(attr) * args.sigma).detach().requires_grad_(True)
                logits      = model(attr_noisy, adj)
                loss_ce     = F.cross_entropy(logits[idx_train], labels[idx_train])
                log_prob    = F.log_softmax(logits[idx_train] / args.temperature, dim=-1)
                total_log_p = log_prob[torch.arange(len(idx_train)), labels[idx_train]].sum()
                grad        = torch.autograd.grad(total_log_p, attr_noisy,
                                                create_graph=True, retain_graph=True)[0]
                loss_dcg    = (grad[idx_train] ** 2).sum(dim=1).mean()
                total_loss  = loss_ce + args.lambda_dcg * loss_dcg
                total_loss.backward()
                ft_optimizer.step()

                if ft_it % 10 == 0:
                    logging.info(f"FT Epoch {ft_it}: CE={loss_ce.item():.4f}  DCG={loss_dcg.item():.4f}")

                del attr_noisy, logits, grad, loss_ce, loss_dcg, total_loss

    # ── evaluate test set ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred     = model(attr, adj)
        test_acc = accuracy(pred, labels, idx_test)
    logging.info(f"Test accuracy: {test_acc:.4f}  (seed={args.seed})")

    # ── save model checkpoint ───────────────────────────────────────────────
    cache_dir = Path(args.artifact_dir) / args.model_type  # ← 加 surrogate/victim 子目录
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{args.dataset}-{args.model}-seed-{args.seed}.pt"
    model_path = cache_dir / model_name

    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved → {model_path}")

    # ── save results JSON ─────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix   = f"train_{args.dataset}_{args.model_type}_{args.model}_seed_{args.seed}"
    existing = glob(str(output_dir / prefix) + "_*.json")
    uid      = (max(int(Path(f).stem.rsplit("_", 1)[-1]) for f in existing) + 1) if existing else 0
    json_path = output_dir / f"{prefix}_{uid:06d}.json"
    json_path.write_text(
        json.dumps(dict(accuracy=test_acc, model_path=str(model_path), config=vars(args)), indent=4)
    )
    logging.info(f"Results saved → {json_path}")


if __name__ == "__main__":
    setup_logging()
    args = parser.parse_args()
    main(args)