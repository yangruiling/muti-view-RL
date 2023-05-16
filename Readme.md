# MUTI-VIEW-RL

# Introduction
基于多粒度对比表征学习的分子属性预测框架

## Getting Started
### Installation

Set up conda environment and clone the github repo

a. create a new environment
```
$ conda create -n muti-view-RL python=3.9.13 -y
$ conda activate muti-view-RL
```

b. Install PyTorch and torchvision(CUDA is required), torch-geometric版本调整[链接](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
```
$ pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

c. Install dependencies
```
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2021.09.1 
$ conda install -c conda-forge tensorboard
$ pip install networkx
$ conda install -c conda-forge nvidia-apex
$ pip install fairseq
```

d. (optional) Install [Apex](https://github.com/NVIDIA/apex)(Pytorch混合精度训练加速器)
```
    1、在https://github.com/kezewang/apex下载apex包，解压至Anaconda3\envs\enes_name\Lisite-packages目录下
    2、使用以下命令安装：
    cd 至 apex 文件夹下，使用Python setup.py install命令安装（推荐使用）
    或 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
    或 pip install -v --no-cache-dir . (without CUDA/C++ extensions)
```
### Run
a. Train model
```
cd MUTI-VIEW-RL
python process.py -n 1000 # data preprocessing
python MVRL.py
```

b. Test pre-train model
```
cd MUTI-VIEW-RL
python process_test.py
python finetune.py
```

### Question
1. pex导入失败：[升级pytorch1.9后出现cannot import name ‘container_abcs‘ from ‘torch._six‘](https://blog.csdn.net/qq_19313495/article/details/120361059)
2. OSError: [WinError 1455] 页面文件太小，无法完成操作。

    解决方案：
    a、num_works 设置为0；
    b、batch_size 调小；

### Pre-trained model optimal weight
- 6500 : ./runs/Feb17_10-57-05
- 9990 ：./runs/Feb17_19-04-17
- 1000 ：./runs/Feb18_15-22-33


## 参考文档：

1. [安装pytorch混合精度计算拓展包-apex](https://www.cxyzjd.com/article/qq_36756866/109579122)
2. [一文讲清楚CUDA、CUDA toolkit、CUDNN、NVCC关系](https://blog.csdn.net/qq_41094058/article/details/116207333)
3. NVIDIA CUDA toolkit完整版本地安装流程

    a、[安装NVIDIA控制面板](https://blog.csdn.net/qq_42772612/article/details/104808171)

    b、[安装CUDA](https://blog.csdn.net/qq_42772612/article/details/104811099)

    c、[安装cuDNN](https://blog.csdn.net/qq_42772612/article/details/104808749)
