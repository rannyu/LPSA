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
We evaluate the proposed method on the ArXiv, MAG, Citeseer, and Pubmed datasets. The code automatically downloads the required dataset to the path specified by --data-dir according to the --dataset argument. 

## Models
We evaluate the proposed method on GCN, SGC, GAT, and GraphSAGE models.
The cache folder contains the checkpoint files for these models.

# Running the Attack
## First Stage: Training the Surrogate Model
Here we take training a surrogate model on ArXiv as an example:

``` bash
python run_train.py --data-dir /path/to/dataset --dataset arxiv --model GCN --seed 0
```

After execution, the surrogate model will be saved to /cache/victim/.

If you prefer to directly use the surrogate model from our paper, we have also provided the pre-trained version in our repository under /cache/victim/.

## Second Stage: Transfer Attack
Here we take attacking SGC on ArXiv as an example:

```bash
python run_train.py --data-dir /path/to/dataset --dataset arxiv --seed 0 --victim_model SGC
```

# License
This source code is made available for research purposes only.

# Acknowledgment
Our code is inspired by [P-RBCD](https://github.com/sigeisler/robustness_of_gnns_at_scale) and [lrGAE](https://github.com/EdisonLeeeee/lrGAE).
Great appreciation for their excellent works.

