# 基于motifs的多模态对比自监督学习框架

## GetStart
### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ Python 3.9.13
$ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
$ torch-geometric版本调整安装链接：https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2021.09.1 
$ conda install -c conda-forge tensorboard


```
