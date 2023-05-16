import os
import csv
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


ATOM_LIST = list(range(1,119))
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


BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]
BONDISRING_LIST = [True, False] # 是否在环中
BONDISAROMATIC_LIST = [True, False] # 是否为芳香键
BONDISCONJUGATED_LIST = [True, False] # 是否为共轭键

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
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


def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                # smiles = row[3]
                smiles = row['smiles']
                label = row[target]
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and label != '':
                    smiles_data.append(smiles)
                    if task == 'classification':
                        labels.append(int(label))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print('Number of data:', len(smiles_data))
    return smiles_data, labels

def read_sdf(data_path, target, task):
    mols = Chem.SDMolSupplier(data_path)
    smiles_data, labels = [], []
    for mol in mols:
        smiles_data.append(mol.GetProp('smile'))
        if task == 'classification':
            labels.append(int(mol.GetProp(target)))
        elif task == 'regression':
            labels.append(float(mol.GetProp(target)))
        else:
            ValueError('task must be either regression or classification')
    print('Number of data:', len(smiles_data))
    return smiles_data, labels, mols



class MolTestDataset(Dataset):
    def __init__(self, data_path, target='p_np', task='classification'):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task

    def __getitem__(self, index):

        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        degree_idx = []
        formal_charge_idx = []
        num_Hs_idx = []
        HybridizationType_idx = []
        atomic_number = []
        
        # 计算原子特征
        for atom in mol.GetAtoms(): # 遍历分子中的每一个原子
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum())) # 存储原子类别序号
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag())) # 获取原子手性类型
            formal_charge_idx.append(FORMAL_CHARGE_LIST.index(atom.GetFormalCharge()))
            degree_idx.append(DEGREE_LIST.index(atom.GetTotalDegree()))
            num_Hs_idx.append(NUM_Hs_LIST.index(atom.GetTotalNumHs()))
            HybridizationType_idx.append(HYBRIDIZATION_LIST.index(atom.GetHybridization()))
            atomic_number.append(atom.GetAtomicNum()) # 存储原子类别序号

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x3 = torch.tensor(formal_charge_idx, dtype=torch.long).view(-1,1)
        x4 = torch.tensor(degree_idx, dtype=torch.long).view(-1,1)
        x5 = torch.tensor(num_Hs_idx, dtype=torch.long).view(-1,1)
        x6 = torch.tensor(HybridizationType_idx, dtype=torch.long).view(-1,1)

        x = torch.cat([x1, x2 ,x3, x4, x5, x6], dim=-1) # 6维原子特征[type_idx, chirality_idx]

        # 计算边特征
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds(): # 遍历化学键
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end] # 记录起始原子和结束原子
            col += [end, start] # 记录反向边
            edge_feat.append([ # 每个化学键被看做两条边，属性为键类型和键方向
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
                BONDISRING_LIST.index(bond.IsInRing()),
                BONDISAROMATIC_LIST.index(bond.GetIsAromatic()),
                BONDISCONJUGATED_LIST.index(bond.GetIsConjugated())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
                BONDISRING_LIST.index(bond.IsInRing()),
                BONDISAROMATIC_LIST.index(bond.GetIsAromatic()),
                BONDISCONJUGATED_LIST.index(bond.GetIsConjugated())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # 获取原子位置矩阵
        AllChem.Compute2DCoords(mol)
        atom_2dcoords = mol.GetConformer().GetPositions()[:, :2]
        pos_2D = torch.tensor(atom_2dcoords, dtype=torch.float64) # dtype=torch.float64 保留位置坐标的的后四位小数

        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.long).view(1,-1)
        elif self.task == 'regression':
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
        data_2D = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, pos=pos_2D)

        return data_2D


    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_path, target, task, max_len):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.max_len = max_len

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
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)
                                
        test_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                  num_workers=self.num_workers, drop_last=False)

        return train_loader, valid_loader, test_loader
