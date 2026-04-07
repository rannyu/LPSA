# LPSA
Official Repository for "LPSA: Leaf-Prior Structural Attack on Graph Neural Networks at Scale".

# Usage
Environment Setup
Before you begin, please ensure that Anaconda or Miniconda is installed on your system. This guide assumes you have a CUDA-enabled GPU.

```bash
# Create and activate a new Conda environment named 'LPSA'
conda create -n LPSA python=3.10
conda activate LPSA

# Install PyTorch
# If you are using a different CUDA version, please refer to the PyTorch website for the appropriate version.
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install PyG
pip install torch_geometric

# Install additional PyG dependencies
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

## Datasets
We evaluate the proposed method on the ArXiv, MAG, Citeseer, and Pubmed datasets.

## Models
We evaluate the proposed method on GCN, SGC, GAT, and GraphSAGE models.
The cache folder contains the checkpoint files for these models.

# Running the Attack
### First Stage: Training the Surrogate Model
```bash
python run_train.py --data-dir /path/to/dataset --dataset arxiv --model GCN --seed 0
```
After execution, the surrogate model will be saved in /cache/victim/.
If you prefer to directly use the surrogate model from our paper, we have provided the pre-trained version in our repository under /cache/victim/
