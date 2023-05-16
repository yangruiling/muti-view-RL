import os
import csv
import math
import time
import signal
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Batch

import networkx as nx
from networkx.algorithms.components import node_connected_component

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
import re
import pickle
from utils.trans_dict import _i2a, _a2i, _pair_list

# 原子特征类型
ATOM_LIST = list(range(1,119)) # 119为最大原子类型序号
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
FORMAL_CHARGE_LIST = [-3, -1, -2, 1, 2, 0, 3]
NUM_Hs_LIST = [0, 1, 2, 3, 4]
HYBRIDIZATION_LIST = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
]
ATOMSYMBOL_LIST = ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other'] # 具有特殊性质的原子类别
IMPLICITVALENCE_LIST = [0, 1, 2, 3, 4, 5, 6] # 原子的隐式化合价
RINGSIZE_LIST = [3, 4, 5, 6, 7, 8] # 几元环
ISHYDROGENDONOR_LIST = [True, False] 
ISHYDROGENACCEPTOR_LIST = [True, False]
ISACIDIC_LIST = [True, False]
ISBASIC_LIST = [True, False]


# 边特征类型
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]
BONDISRING_LIST = [True, False] # 是否在环中
BONDISAROMATIC_LIST = [True, False] # 是否为芳香键
BONDISCONJUGATED_LIST = [True, False] # 是否为共轭键

class TimeoutError(Exception):
    pass


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data

def get_tokenizer_re(atoms):
    return re.compile("(" + "|".join(atoms) + r"|\%\d\d|.)") # re.compile("正则表达式") 

def smiles_separator(smile, _pair_list=_pair_list):
    """
    :param _pair_list: the two-atom list to recognize
    :param smiles: str, the initial smiles string
    :return: list, the smiles list after seperator
                    [recognize the atom and the descriptor]
    """
    if _pair_list:
        reg = get_tokenizer_re(_pair_list)
    else:
        reg = get_tokenizer_re(_pair_list) 

    smiles_list = reg.split(smile)[1::2] # list[start:end:step],此操作去除空元素
    return smiles_list

def smile_to_tensor(smile: str, pad_size=50):
    # change smiles[text] into tensor format
    smile_tensor = smiles_separator(smile)
    cur_smiles_len = len(smile_tensor)
    res = [_a2i[s] for s in smile_tensor]
    if cur_smiles_len > pad_size:
        res = res[:pad_size]
    else:
        res = res + [0] * (pad_size - cur_smiles_len)
    smile_tensor = torch.tensor(res).long()
    return smile_tensor, cur_smiles_len

'''
description: 
param {*} m = 2D分子对象
param {*} algo = MMFF/ETKDG/UFF: Redit中生成3D像的几种方法
return {*}
'''
def optimize_conformer(m, algo="MMFF"):
    print("Calculating {} ...".format(Chem.MolToSmiles(m)))

    mol = Chem.AddHs(m) # 显式化氢原子

    if algo == "ETKDG":
        # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        if k != 0:
            return None, None, None

    elif algo == "UFF":
        # Universal Force Field
        # 重复生成P次坐标以解决生成的3D坐标不准确的问题
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5) # 生成50个3D构象，保存在mol分子中
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None, None

        if not arr:
            return None, None, None

        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = Chem.Conformer(mol.GetConformers()[int(idx)]) # 访问第idx个构像
            mol.RemoveAllConformers() # 删除所有构象
            mol.AddConformer(conf) #添加已存储的第idx个构象

    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5) # 生成50个3D构象，保存在mol分子中
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None, None

        if not arr:
            return None, None, None

        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=20000) # 使用MMFF力场优化分子构像
            idx = np.argmin(arr, axis=0)[1]
            conf =Chem.Conformer(mol.GetConformers()[int(idx)])  
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    mol_3D = Chem.RemoveHs(mol)
    return mol_3D

def index_function(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]
    return allowable_set.index(x)

class MoleculeDataset(Dataset):
    # def __init__(self, mols_data, smiles_data, smiles_len_data , algo, max_len):
    def __init__(self, mols_data, algo, max_len):
        super(Dataset, self).__init__()
        self.mols_data = mols_data
        # self.smiles_data = smiles_data
        # self.smiles_len_data = smiles_len_data
        self.algo = algo
        self.max_len = max_len

    def __getitem__(self, index):
        '''
        description: 数据处理部分
        return {data_2D = 2D分子图, data_3D = 3D分子图, mol = 3D分子对象, N = 原子数, frag_mols = 分子片段, frag_indices = 片段序号}
        '''
        mols_i = self.mols_data[index]
        Seq_feature = mols_i['Seq_feature']
        Seq_len = mols_i['Seq_len']
        data_2D = mols_i['data_2D']
        Gdata_2D = Data(x=data_2D['x'], edge_index=data_2D['edge_index'], edge_attr=data_2D['edge_attr'], pos=data_2D['pos_2D'])
        data_3D = mols_i['data_3D']
        Gdata_3D = Data(x=data_3D['x'], edge_index=data_3D['edge_index'], edge_attr=data_3D['edge_attr'], pos=data_3D['pos_3D'])

        # data_2D(x=[N, 6], edge_index=[2, M*2], edge_attr=[M*2, 5], pos = [N, 2])
        # data_3D(x=[N, 13], edge_index=[2, M*2], edge_attr=[M*2, 5], pos = [N, 3])
        mol = mols_i['mol']
        N = mols_i['N']
        frag_mols = mols_i['frag_mols']
        frag_indices = mols_i['frag_indices']

        return Seq_feature, Seq_len, Gdata_2D, Gdata_3D, mol, N, frag_mols, frag_indices
    
    def __len__(self):
        return len(self.mols_data)


def collate_fn(batch):
    Seq_feature, Seq_len, data_2D, data_3D, mols, atom_nums, frag_mols, frag_indices = zip(*batch)

    frag_mols = [j for i in frag_mols for j in i] # 将所有官能团中的片段合并为一个大list

    data_2D = Batch.from_data_list(data_2D) # 将同一批次的样本拼成一个tensor
    data_3D = Batch.from_data_list(data_3D)
    Seq_feature = torch.stack(Seq_feature, dim=0)
    Seq_len = torch.tensor(Seq_len, dtype=torch.long)
    # data_2D(x=[N, 6], edge_index=[2, M*2], edge_attr=[M*2, 5], pos = [N, 2])
    # data_3D(x=[N, 13], edge_index=[2, M*2], edge_attr=[M*2, 5], pos = [N, 3])

    # 为当前batch中的所有原子创建索应，原子在第几个官能团中，则对应位置为几，不在官能团中的原子对应位置为0
    data_2D.motif_batch = torch.zeros(data_2D.x.size(0), dtype=torch.long)  
    data_3D.motif_batch = torch.zeros(data_3D.x.size(0), dtype=torch.long)

    curr_indicator = 1
    curr_num = 0
    for N, indices in zip(atom_nums, frag_indices): # N：当前batch中某个样本的原子数；indices：当前batch中某个样本官能团索引列表
        for idx in indices: # 取出某个官能团包含的全部原子索引
            curr_idx = np.array(list(idx)) + curr_num
            data_2D.motif_batch[curr_idx] = curr_indicator
            data_3D.motif_batch[curr_idx] = curr_indicator
            curr_indicator += 1
        curr_num += N

    return Seq_feature, Seq_len, data_2D, data_3D, mols, frag_mols

def load(path):
        with open(path, 'rb') as f:
            smile_tensor_list, cur_smiles_len_list = pickle.load(f)
        return smile_tensor_list, cur_smiles_len_list

class MoleculeDatasetWrapper(object):
    def __init__(self, max_len, batch_size, num_workers, valid_size, data_path, algo):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.algo = algo
        self.max_len = max_len

    # 直接载入Tensor数据
    def get_data_loaders(self):
        print("Load Mols files: ~~~~~~~~~~")
        mol_list = {}
        index = 0
        with open(self.data_path,'rb') as f:
            while True:
                try:
                    mol = pickle.load(f)
                    mol_list[index]= mol
                    index += 1
                except EOFError:
                    break

        num_train = len(mol_list) # 数据集中分子的数量
        indices = list(range(num_train)) # 分子对应序号
        np.random.shuffle(indices) 

        split = int(np.floor(self.valid_size * num_train))

        train_idx, valid_idx = indices[split:], indices[:split] # 使用序号划分训练集和验证集
        # train_idx, valid_idx = indices[:], indices[:]
        
        train_mols = [mol_list[i] for i in train_idx]
        valid_mols = [mol_list[i] for i in valid_idx]
        del mol_list
        del indices
        print("训练集分子数: " + str(len(train_mols)) + "; 验证集分子数："+ str(len(valid_mols)))

        train_dataset = MoleculeDataset(train_mols, self.algo, self.max_len) #使用Dataset类封装数据集
        valid_dataset = MoleculeDataset(valid_mols, self.algo, self.max_len) 

        train_loader = DataLoader( # 定义数据读取器
            train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
            num_workers=self.num_workers, drop_last=True
        )

        return train_loader, valid_loader



if __name__ == "__main__":
    # 载入初始数据集
    dataset = MoleculeDatasetWrapper(500, 20, 30, 0.05,'/home/yrl/muti-view-RL/data/pubchem-10m-clean-1000.pkl', "MMFF")
    # 1.调用MoleculeDataset类处理原始数据集
    # 2.调用torch.util.data.dataloader.collate_fn()拼接数据list生成mini-batch Tensor
    train_loader, valid_loader = dataset.get_data_loaders() 
    for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols, frag_mols) in enumerate(train_loader):
        print("BN: ", bn, Seq_len)
   
   
    # 载入数据（Pytorch在载入数据时才会执行相关的处理命令）
    # for bn, (Seq_feature, Seq_len, g1, g2, mols, frag_mols) in enumerate(train_loader):
    #     print(Seq_feature)
    #     print(Seq_len)
    #     print(g1)
    #     print(g2)
    #     print(mols)
    #     print(frag_mols)
    # # mol3d = optimize_conformer( Chem.MolFromSmiles("CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1"), "MMFF")
    # # mol3d = optimize_conformer( Chem.MolFromSmiles("CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1"), "MMFF")
    # s = smile_to_tensor("CCCCCC1=NOCCC1F")
    # print(s)






    # 载入3D分子数据，边训练边处理
    # def get_data_loaders(self):
    #     print("Load Mols files: ~~~~~~~~~~")
    #     mol = Chem.SDMolSupplier(self.sdf_data_path) # 从sdf文件中读取分子
    #     mols = [mol[i] for i in range(3000)]
    #     # smiles_data = read_smiles(self.data_path)
    #     # smile_tensor_list, cur_smiles_len_list = load('E:\Project\muti-view-RL\data\est.pkl') # 载入编码好的SMILEs Tensor

    #     num_train = len(mols) # 数据集中分子的数量
    #     indices = list(range(num_train)) # 分子对应序号
    #     np.random.shuffle(indices) 

    #     split = int(np.floor(self.valid_size * num_train))

    #     train_idx, valid_idx = indices[split:], indices[:split] # 使用序号划分训练集和验证集
    #     # train_idx, valid_idx = indices[:], indices[:]
        
    #     train_mols = [mols[i] for i in train_idx]
    #     # train_smiles = [smile_tensor_list[i] for i in train_idx]
    #     # train_smiles_len = [cur_smiles_len_list[i] for i in train_idx]
    #     valid_mols = [mols[i] for i in valid_idx]
    #     # valid_smiles = [smile_tensor_list[i] for i in valid_idx]
    #     # valid_smiles_len = [cur_smiles_len_list[i] for i in valid_idx]

    #     del mols
    #     # del smile_tensor_list
    #     # del cur_smiles_len_list
    #     print("训练集分子数: " + str(len(train_mols)) + "; 验证集分子数："+ str(len(valid_mols)))

    #     # train_dataset = MoleculeDataset(train_mols, train_smiles, train_smiles_len, self.algo, self.max_len) #使用Dataset类封装数据集
    #     # valid_dataset = MoleculeDataset(valid_mols, valid_smiles, valid_smiles_len, self.algo, self.max_len) 
    #     train_dataset = MoleculeDataset(train_mols, self.algo, self.max_len) #使用Dataset类封装数据集
    #     valid_dataset = MoleculeDataset(valid_mols, self.algo, self.max_len) 

    #     train_loader = DataLoader( # 定义数据读取器
    #         train_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
    #         num_workers=self.num_workers, drop_last=True, shuffle=True
    #     )
    #     valid_loader = DataLoader(
    #         valid_dataset, batch_size=self.batch_size, collate_fn=collate_fn,
    #         num_workers=self.num_workers, drop_last=True
    #     )

    #     return train_loader, valid_loader
