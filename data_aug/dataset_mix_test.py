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
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import re
import pickle
from utils.trans_dict import _i2a, _a2i, _pair_list
import timeout_decorator

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


def read_mols(data_path, target, task):
    print("Load Mols files: ~~~~~~~~~~")
    mol_list = {}
    index = 0
    # with open('/home/yrl/muti-view-RL/data/freesolv/expt.pkl','rb') as f:
    with open(data_path,'rb') as f:
        while True:
            try:
                mol = pickle.load(f)
                mol_list[index]= mol
                index += 1
            except EOFError:
                break
    return mol_list
'''
description: 
param {*} m = 2D分子对象
param {*} algo = MMFF/ETKDG/UFF: Redit中生成3D像的几种方法
return {*}
'''
class MolTestDataset(Dataset):
    # def __init__(self, mols_data, smiles_data, smiles_len_data , algo, max_len):
    def __init__(self, data_path, target='p_np', task='classification', algo = 'MMFF' , max_len = 500):
        super(Dataset, self).__init__()
        self.mols_data = read_mols(data_path, target, task)
        # self.smiles_data, self.labels = read_smiles('/home/yrl/muti-view-RL/data/bbbp/raw/BBBP.csv', target, task)
        self.task = task
        self.algo = algo
        self.max_len = max_len
    
    @timeout_decorator.timeout(240)
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
        target = mols_i['label']

        return Seq_feature, Seq_len, Gdata_2D, Gdata_3D, mol, N, target
    
    def __len__(self):
        return len(self.mols_data)


def collate_fn(batch):
    Seq_feature, Seq_len, data_2D, data_3D, mols, atom_nums, target= zip(*batch)
    data_2D = Batch.from_data_list(data_2D) # 将同一批次的样本拼成一个tensor
    data_3D = Batch.from_data_list(data_3D)
    Seq_feature = torch.stack(Seq_feature, dim=0)
    Seq_len = torch.tensor(Seq_len, dtype=torch.long)
    target = torch.tensor(target)

    return Seq_feature, Seq_len, data_2D, data_3D, mols, target

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold
def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind in range(data_len):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(dataset.mols_data[ind]['smile'])
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    
    print('train: {}, valid: {}, test: {}'.format(
        len(train_inds), len(valid_inds), len(test_inds)))
    return train_inds, valid_inds, test_inds


class MoleculeTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, algo, data_path, save_path ,target, task, max_len):
        super(object, self).__init__()
        self.data_path = save_path + target + '.pkl'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.algo = algo
        self.target = target
        self.task = task
        self.max_len = max_len

    # 直接载入Tensor数据
    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task, max_len=self.max_len)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, collate_fn=collate_fn, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, collate_fn=collate_fn, drop_last=True)
                                
        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                  num_workers=self.num_workers, collate_fn=collate_fn, drop_last=True)

        return train_loader, valid_loader, test_loader



if __name__ == "__main__":
    # 载入初始数据集
    dataset = MoleculeTestDatasetWrapper(20, 24, 0.1,0.1, "MMFF",'/home/yrl/muti-view-RL/data/bbbp/raw/BBBP.csv', "p_np", 'classification', 234)
    # 1.调用MoleculeDataset类处理原始数据集
    # 2.调用torch.util.data.dataloader.collate_fn()拼接数据list生成mini-batch Tensor
    train_loader, valid_loader, test_loader = dataset.get_data_loaders() 
    for bn, (Seq_feature, Seq_len, g_2d, g_3d, mols) in enumerate(train_loader):
        print("BN: ", bn, Seq_len)




