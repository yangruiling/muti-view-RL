import csv
import yaml
import os
import numpy as np
import multiprocessing as mp
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.BRICS import BRICSDecompose, FindBRICSBonds, BreakBRICSBonds
import torch
import re
from utils.trans_dict import _i2a, _a2i, _pair_list
import pickle
from tqdm import *
# from tqdm.notebook import tqdm
import networkx as nx
from networkx.algorithms.components import node_connected_component
from torch_scatter import scatter
import gc
import argparse
import timeout_decorator

# import sys


# 原子特征类型
ATOM_LIST = list(range(1,119)) # 119为最大原子类型序号
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
FORMAL_CHARGE_LIST = [-3, -1, -2, 1, 2, 0, 3] # 原子电荷
NUM_Hs_LIST = [0, 1, 2, 3, 4] # H原子数量
HYBRIDIZATION_LIST = [  # 杂化轨道类型
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
ISHYDROGENDONOR_LIST = [True, False]  # 某原子是否是氢供体
ISHYDROGENACCEPTOR_LIST = [True, False] # 某原子是否是氢受体
ISACIDIC_LIST = [True, False] # 酸性
ISBASIC_LIST = [True, False] # 是否是基础原子


# 边特征类型
BOND_LIST = [ 
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [ # 边的方向
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
    Chem.rdchem.BondDir.EITHERDOUBLE
]
BONDISRING_LIST = [True, False] # 是否在环中
BONDISAROMATIC_LIST = [True, False] # 是否为芳香键
BONDISCONJUGATED_LIST = [True, False] # 是否为共轭键

num_process = 1 #mp.cpu_count()-24 

def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with tqdm(total=9999918) as pbar:
            for i, row in enumerate(csv_reader):
                    smiles = row[-1]
                    smiles_data.append(smiles)
                    pbar.update(1)
    return smiles_data

def read_smiles_1(data_path, start_index, end_index):
    smiles_data = []
    index = range(start_index, end_index)
    total = len(index)
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        with tqdm(total=total) as pbar:
            for i, row in enumerate(csv_reader):
                if i < start_index:
                    continue
                elif i in index:
                    smiles = row[-1]
                    smiles_data.append(smiles)
                    pbar.update(1)
                else:
                    break
    return smiles_data

def optimize_conformer(idx, smi, m, algo="MMFF"):
#    if idx%100==0:
#        print("Calculating {}: {} ...".format(idx, Chem.MolToSmiles(m)))
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
    return smi, mol_3D

def optimize_conformer_star(args):
    return optimize_conformer(*args)

def converter(smiles_data, target_path, process=num_process, algo='MMFF'): # 转换器
    # index = [0]
    # index = range(0, 10000)
    for i in tqdm(range(0, 1), desc='Processing'):
        if i !=9999:
            mols = []
            start_index = int(i*1000) # int(i*100000)  
            end_index = int((i+1)*1000) #int((i+1)*100000)
            print("当前进度：" + str(start_index) + " ~ " + str(end_index))
            smi = smiles_data[start_index:end_index]
            for smile in smi: #
                mol = Chem.MolFromSmiles(smile)
                mols.append(mol)
            mol_idx = list(range(len(mols)))
            algo = [algo]*len(mols)
            print("Loaded {} Molecules".format(len(mols)))
        else:
            mols = []
            print("process last Molecules")
            smi = smiles_data[i*1000:]
            for smile in smi:
                mol = Chem.MolFromSmiles(smile)
                mols.append(mol)
            mol_idx = list(range(len(mols)))
            algo = [algo]*len(mols)
            print("Loaded {} Molecules".format(len(mols)))

        # Optimize coordinate using multiprocessing
        print("Optimizing Conformers..., num_worker = {}".format(process))
        
        # 多进程处理分子
        # pool = mp.Pool(process) #process
        # args = zip(mol_idx, smi, mols, algo)
        # results = pool.starmap(optimize_conformer, args)
        # results = list(pool.imap(optimize_conformer_star, args))
        # pool.close()
        # results = pool.starmap(optimize_conformer, zip(mol_idx, smiles_data, mols, algo))

        # 按顺序逐个处理分子
        results = []
        for j in tqdm(range(0,40), desc='Processing'):
            smi_i, mol_3d_i = optimize_conformer(mol_idx[j], smiles_data[j], mols[j], algo[j])
            results.append([smi_i, mol_3d_i])
        del mols
        

        # Remove None and add properties
        mol_3d_list = []
        for result in results:
            if result[0] != None:
                smi_p = result[0]
                mol_3d = result[1]
                mol_3d.SetProp("smile", str(smi_p))
                mol_3d_list.append(mol_3d) 
        print("{} Molecules Optimized".format(len(mol_3d_list)))
        del results

        # Save molecules
        print("Saving File...")
        path = target_path + "/pubchem-10m-clean-" + str(i) +".sdf"
        w = Chem.SDWriter(path)
        for m in mol_3d_list:
            w.write(m)
        print("Saved {} Molecules to {}".format(len(mol_3d_list), path))
        del mol_3d_list

def get_tokenizer_re(atoms):
    return re.compile("(" + "|".join(atoms) + r"|\%\d\d|.)") # re.compile("正则表达式") 

def smiles_separator(smile, pair_list=_pair_list):
    """
    :param pair_list: the two-atom list to recognize
    :param smiles: str, the initial smiles string
    :return: list, the smiles list after seperator
                    [recognize the atom and the descriptor]
    """
    if pair_list:
        reg = get_tokenizer_re(pair_list)
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

def index_function(x, allowable_set):
    # If x is not in allowed set, use last index
    if x not in allowable_set:
        x = allowable_set[-1]
    return allowable_set.index(x)

@timeout_decorator.timeout(3)
def converter_data_to_tensor(save_path, i , mol_3D):
        # 编码序列特征
    smile = mol_3D.GetProp('smile') # 获取分子式
    Seq_feature, Seq_len = smile_to_tensor(smile, pad_size=500) # 将分子式编码为Tensor特征 max_len = 500
    # Seq_feature = self.smiles_data[index]
    # Seq_len = self.smiles_len_data[index]
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
    mol = Chem.MolFromSmiles(smile) # 将数据从SMILES转化为2D分子
    
    
    Chem.SanitizeMol(mol)
    
    
    mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms() # 获取原子数
    M = mol.GetNumBonds() # 获取化学键数

    type_idx = []
    chirality_idx = []
    degree_idx = []
    formal_charge_idx = []
    num_Hs_idx = []
    HybridizationType_idx = []
    atomic_number = []
    
    # 计算原子特征
    # for atom in mol.GetAtoms(): # 遍历分子中的每一个原子
    for atom_idx in range(N):
        atom = mol.GetAtomWithIdx(atom_idx)
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
    # pos = torch.tensor(atom_2dcoords, dtype=torch.long)

    # 构建2D分子图数据
    data_2D = {"x":x, "edge_index":edge_index, "edge_attr":edge_attr, "pos_2D":pos_2D}

    # try:
    #     frag_mols, frag_indices = get_fragments(mol)
    # except Exception as e:
    #     print ('Timeout Error Catched!')
    #     print (e)
    #     return "-1" 
    frag_mols, frag_indices = get_fragments(mol)
    
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
    # 读取3D分子
    mol = Chem.AddHs(mol_3D)
    
    #定义分子间作用力 - 氢键、肽键...
    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]") # 定义氢键供体
    hydrogen_acceptor = Chem.MolFromSmarts( # 定义氢键受体
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts(
    "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol)
    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ()) # 元组内部数字求和
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())

    ring = mol.GetRingInfo()

    atomsymbol_idx = []
    degree_idx = []
    HybridizationType_idx = []
    implicitvalence_idx = []
    formal_charge_idx = []
    atomic_number = []
    inringsize_idx = []
    num_Hs_idx = []
    chirality_idx = []
    ishydorogendonor_idx = []
    ishydorogenaccector_idx = []
    isacidic_idx = []
    isbasic_idx = []


    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)

        atomsymbol_idx.append( index_function(atom.GetSymbol(), ATOMSYMBOL_LIST)) # 获取有特殊性质的原子类别
        degree_idx.append( index_function(atom.GetTotalDegree(), DEGREE_LIST))
        HybridizationType_idx.append( index_function(atom.GetHybridization(), HYBRIDIZATION_LIST))
        implicitvalence_idx.append( index_function(atom.GetImplicitValence(), IMPLICITVALENCE_LIST))
        formal_charge_idx.append( index_function(atom.GetFormalCharge(), FORMAL_CHARGE_LIST))
        atomic_number.append(atom.GetAtomicNum()) # 存储原子类别序号
        num_Hs_idx.append( index_function(atom.GetTotalNumHs(), NUM_Hs_LIST))
        
        if ring.IsAtomInRingOfSize(atom_idx, 3):
            inringsize_idx.append(RINGSIZE_LIST.index(3))
        elif ring.IsAtomInRingOfSize(atom_idx, 4):
            inringsize_idx.append(RINGSIZE_LIST.index(4))
        elif ring.IsAtomInRingOfSize(atom_idx, 5):
            inringsize_idx.append(RINGSIZE_LIST.index(5))
        elif ring.IsAtomInRingOfSize(atom_idx, 6):
            inringsize_idx.append(RINGSIZE_LIST.index(6))
        elif ring.IsAtomInRingOfSize(atom_idx, 7):
            inringsize_idx.append(RINGSIZE_LIST.index(7))
        elif ring.IsAtomInRingOfSize(atom_idx, 8):
            inringsize_idx.append(RINGSIZE_LIST.index(8))
        else:
            inringsize_idx.append(6)

        CIPCode = ["R", "S", "others"]
        try:
            idx = CIPCode.index(atom.GetProp('_CIPCode'))
            chirality_idx.append(idx)
        except:
            chirality_idx.append( index_function("others", CIPCode))
        
        ishydorogendonor_idx.append(ISHYDROGENDONOR_LIST.index(atom_idx in hydrogen_donor_match))
        ishydorogenaccector_idx.append(ISHYDROGENACCEPTOR_LIST.index(atom_idx in hydrogen_acceptor_match))
        isacidic_idx.append(ISACIDIC_LIST.index(atom_idx in acidic_match))
        isbasic_idx.append(ISBASIC_LIST.index(atom_idx in basic_match))

    x1 = torch.tensor(atomsymbol_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(degree_idx, dtype=torch.long).view(-1,1)
    x3 = torch.tensor(HybridizationType_idx, dtype=torch.long).view(-1,1)
    x4 = torch.tensor(implicitvalence_idx, dtype=torch.long).view(-1,1)
    x5 = torch.tensor(formal_charge_idx, dtype=torch.long).view(-1,1)
    x6 = torch.tensor(atomic_number, dtype=torch.long).view(-1,1)
    x7 = torch.tensor(num_Hs_idx, dtype=torch.long).view(-1,1)
    x8 = torch.tensor(inringsize_idx, dtype=torch.long).view(-1,1)
    x9 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x10 = torch.tensor(ishydorogendonor_idx, dtype=torch.long).view(-1,1)
    x11 = torch.tensor(ishydorogenaccector_idx, dtype=torch.long).view(-1,1)
    x12 = torch.tensor(isacidic_idx, dtype=torch.long).view(-1,1)
    x13 = torch.tensor(isbasic_idx, dtype=torch.long).view(-1,1)


    x = torch.cat([x1, x2 ,x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13], dim=-1) # 13维原子特征[type_idx, chirality_idx]

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

    pos_matrix = mol.GetConformer().GetPositions()
    pos_3D = torch.tensor(pos_matrix, dtype=torch.float64) # 保留位置坐标的的后四位小数
    # 构建3D分子图数据
    data_3D = { "x":x, "edge_index":edge_index, "edge_attr":edge_attr, "pos_3D":pos_3D }

    # data_2D(x=[N, 6], edge_index=[2, M*2], edge_attr=[M*2, 5], pos = [N, 2])
    # data_3D(x=[N, 13], edge_index=[2, M*2], edge_attr=[M*2, 5], pos = [N, 3])

    f_list = {"Seq_feature": Seq_feature, "Seq_len": Seq_len, "data_2D": data_2D, "data_3D": data_3D, "mol": mol, "N": N, "frag_mols": frag_mols, "frag_indices": frag_indices}
    with open(save_path, 'ab') as fo: # wb是覆盖写，如果需要追加，则为'ab' 
        pickle.dump(f_list, fo)
    
    return "1"
    
def get_fragment_indices(mol):
    '''
    description: BRICS拆解分子得到分子官能团
    return {index:使用原子序号作为索引的motif元组, 无重复}
    '''    
    bonds = mol.GetBonds()
    edges = []
    for bond in bonds:
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    molGraph = nx.Graph(edges)

    BRICS_bonds = list(FindBRICSBonds(mol))
    break_bonds = [b[0] for b in BRICS_bonds]
    break_atoms = [b[0][0] for b in BRICS_bonds] + [b[0][1] for b in BRICS_bonds]
    molGraph.remove_edges_from(break_bonds)

    indices = []
    for atom in break_atoms:
        n = node_connected_component(molGraph, atom)  # 返回包含某个节点的最大团，在此处即某个官能团中包含的节点序列
        if len(n) > 3 and n not in indices:
            indices.append(n)
    indices = set(map(tuple, indices))
    return indices

def get_fragments(mol):
    ref_indices = get_fragment_indices(mol)

    frags = list(BRICSDecompose(mol, returnMols=True))
    mol2 = BreakBRICSBonds(mol)

    extra_indices = []
    for i, atom in enumerate(mol2.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            extra_indices.append(i)
    extra_indices = set(extra_indices)

    frag_mols = []
    frag_indices = []
    for frag in frags: # 筛选解离后的片段
        indices = mol2.GetSubstructMatches(frag)
        # if len(indices) >= 1:
        #     idx = indices[0]
        #     idx = set(idx) - extra_indices
        #     if len(idx) > 3:
        #         frag_mols.append(frag)
        #         frag_indices.append(idx)
        if len(indices) == 1:
            idx = indices[0]
            idx = set(idx) - extra_indices
            if len(idx) > 3:
                frag_mols.append(frag)
                frag_indices.append(idx)
        else:
            for idx in indices:
                idx = set(idx) - extra_indices
                if len(idx) > 3:
                    for ref_idx in ref_indices:
                        if (tuple(idx) == ref_idx) and (idx not in frag_indices):
                            frag_mols.append(frag)
                            frag_indices.append(idx)

    return frag_mols, frag_indices

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('-n', help='运行次数')
args = parser.parse_args()

if __name__ == "__main__":
    # out_path = './output/a_' + args.n + '.log'
    # f = open(out_path, 'a')
    # sys.stdout = f
    # sys.stderr = f
    path = "/home/yrl/muti-view-RL/data/sdf/pubchem-10m-clean-"+ args.n +".sdf"
    save_path = "/home/yrl/muti-view-RL/data/pkl/pubchem-10m-clean-" + args.n + ".pkl"
    mols_3D = Chem.SDMolSupplier(path)
    # converter_data_to_tensor(mols_3D)    
    index = range(838000,len(mols_3D))
    for i in tqdm(index, desc= args.n, mininterval = 5.0):
    # for i in tqdm(index, desc= args.n):
        try:
            converter_data_to_tensor(save_path, i, mols_3D[i])
        except Exception as e:
            print ('Timeout Error Catched!')
            print (e)
            continue
        
        # if m == "-1":
        #     continue

    # 6000 - 838000
    # # 统一读取数据
    # smiles_data = read_smiles("/home/yrl/muti-view-RL/data/pubchem-10m-clean.txt")
    # converter(smiles_data, "/home/yrl/muti-view-RL/data/sdf" ) # 转化为3d构象
    # del smiles_data
   
    # 数据合并
    # mols = []
    # for i in tqdm(range(3250, 6500)): # 0~3007(2999968)、3250～6500(3040966)、6500～9990(3488930)
    #     path = "/home/yrl/muti-view-RL/data/sdf/pubchem-10m-clean-" + str(i)+ ".sdf"
    #     if(os.path.exists(path)):
    #         mols.extend(Chem.SDMolSupplier(path))
    #     else:
    #         continue
    # # Save molecules
    # print("Saving File...")
    # path = "/home/yrl/muti-view-RL/data/pubchem-10m-clean-6500.sdf"
    # w = Chem.SDWriter(path)
    # for m in tqdm(mols):
    #     w.write(m)
    # print("Saved {} Molecules to {}".format(len(mols), path))

    # # 数据拆分
    # mols = []
    # path = "/home/yrl/muti-view-RL/data/pubchem-10m-clean-9990.sdf"
    # if(os.path.exists(path)):
    #     mols.extend(Chem.SDMolSupplier(path))
    # # Save molecules
    # print("Saving File 1...")
    # path = "/home/yrl/muti-view-RL/data/sdf/pubchem-10m-clean-7500.sdf"
    # w = Chem.SDWriter(path)
    # for m in tqdm(mols[0:1000000]):
    #     w.write(m)
    # print("Saved {} Molecules to {}".format(len(mols), path))


   