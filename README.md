# Medical Abbreviation Resolution via Knowledge Enhanced Tranformer

## Python Environment Setup

1. conda environment
```
conda create --name=medar python=3.10
conda activate medar
```

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -r requirements.txt
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
ipython kernel install --user --name=medar
```

Exit and reopen a terminal (conda env medar)

```
jupyter lab --no-browser --port=8888
```

## Dataset Setup

Dataset
- MeDAL

```
# MedAL dataset is avaliable on HuggingFace dataset.
import datasets
...
```