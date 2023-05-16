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

num_process = 1 #mp.cpu_count()-24 

def optimize_conformer_star(args):
    return optimize_conformer(*args)

def get_config():
    config = yaml.load(open("/home/yrl/muti-view-RL/config_finetune.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)
    # config = yaml.load(open("config_finetune.yaml", "r", encoding='utf-8'), Loader=yaml.FullLoader)

    if config['task_name'] == 'BBBP':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/bbbp/raw/BBBP.csv'
        config['dataset']['save_path'] = './data/bbbp/'
        config['dataset']['max_len'] = 234
        target_list = ["p_np"]

    elif config['task_name'] == 'Tox21':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/tox21/tox21.csv'
        config['dataset']['save_path'] = './data/tox21/'
        config['dataset']['max_len'] = 267
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif config['task_name'] == 'ClinTox':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/clintox/clintox.csv'
        config['dataset']['save_path'] = './data/clintox/'
        config['dataset']['max_len'] = 254
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif config['task_name'] == 'HIV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/hiv/HIV.csv'
        config['dataset']['save_path'] = './data/hiv/'
        config['dataset']['max_len'] = 495
        target_list = ["HIV_active"]

    elif config['task_name'] == 'BACE':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/bace/bace.csv'
        config['dataset']['save_path'] = './data/bace/'
        config['dataset']['max_len'] = 194
        target_list = ["Class"]

    elif config['task_name'] == 'SIDER':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/sider/sider.csv'
        config['dataset']['save_path'] = './data/sider/'
        config['dataset']['max_len'] = 982
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", 
            "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", 
            "Immune system disorders", "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", 
            "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", 
            "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", 
            "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif config['task_name'] == 'MUV':
        config['dataset']['task'] = 'classification'
        config['dataset']['data_path'] = './data/muv/muv.csv'
        config['dataset']['save_path'] = './data/muv/'
        config['dataset']['max_len'] = 84
        target_list = [
            "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-692", "MUV-712", "MUV-713", 
            "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
        ]

    elif config['task_name'] == 'FreeSolv':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/freesolv/freesolv.csv'
        config['dataset']['save_path'] = './data/freesolv/'
        config['dataset']['max_len'] = 500
        target_list = ["expt"]
    
    elif config["task_name"] == 'ESOL':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/esol/esol.csv'
        config['dataset']['save_path'] = './data/esol/'
        config['dataset']['max_len'] = 500
        target_list = ["measured log solubility in mols per litre"]

    elif config["task_name"] == 'Lipo':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/lipophilicity/Lipophilicity.csv'
        config['dataset']['save_path'] = './data/lipophilicity/'
        config['dataset']['max_len'] = 500
        target_list = ["exp"]

    elif config["task_name"] == 'qm7':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm7/qm7.csv'
        config['dataset']['save_path'] = './data/qm7/'
        config['dataset']['max_len'] = 500
        target_list = ["u0_atom"]

    elif config["task_name"] == 'qm8':
        config['dataset']['task'] = 'regression'
        config['dataset']['data_path'] = './data/qm8/qm8.csv'
        config['dataset']['save_path'] = './data/qm8/'
        config['dataset']['max_len'] = 500
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", 
            "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]

    else:
        raise ValueError('Unspecified dataset!')

    print(config)
    return config, target_list

def optimize_conformer(m, algo="MMFF"):
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
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
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
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000) # 使用MMFF力场优化分子构像
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

@timeout_decorator.timeout(20)
def converter_data_to_tensor(smile, label_list, config):

    Seq_feature, Seq_len = smile_to_tensor(smile, pad_size=config['dataset']['max_len']) # 将分子式编码为Tensor特征 max_len = 500

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
    
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
    # 读取3D分子
    mol_3D = optimize_conformer(mol)
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
    data_3D = { "x":x, "edge_index":edge_index, "edge_attr":edge_attr, "pos_3D":pos_3D}

    

    for i, target in enumerate(config['dataset']['target']):
        save_path = os.path.join(config['dataset']['save_path'], target + '.pkl') 
        if label_list[i] != -1:
            f_list = {"smile": smile, "Seq_feature": Seq_feature, "Seq_len": Seq_len, "data_2D": data_2D, "data_3D": data_3D, "mol": mol, "N": N, "label": label_list[i]}
            with open(save_path, 'ab') as fo: # wb是覆盖写，如果需要追加，则为'ab' 
                pickle.dump(f_list, fo)


    
    

def read_smiles(data_path, target, task):
    smiles_data, labels = [], []
    # with open('/home/yrl/muti-view-RL/data/freesolv/freesolv.csv') as csv_file:
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

def read_smiles_muti(data_path, target_list, task):
    smiles_data, label_list = [], []
    # with open('/home/yrl/muti-view-RL/data/clintox/clintox.csv') as csv_file:
    with open(data_path) as csv_file:
        # csv_reader = csv.reader(csv_file, delimiter=',')
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i != 0:
                # smiles = row[3]
                smiles = row['smiles']
                labels = []
                mol = Chem.MolFromSmiles(smiles)
                if mol != None and task == 'classification':
                    for target in target_list:
                        if row[target] != '':
                            labels.append(int(row[target]))
                        else:
                            labels.append(-1)
                    smiles_data.append(smiles)
                    label_list.append(labels)
                elif mol != None and task == 'regression':
                    for target in target_list:
                        if row[target] != '':
                            labels.append(float(row[target]))
                        else:
                            labels.append(-1)
                    smiles_data.append(smiles)
                    label_list.append(labels)
                else:
                    ValueError('task must be either regression or classification')
                
    print('Number of data:', len(smiles_data))
    return smiles_data, label_list

if __name__ == '__main__':
    config, target_list = get_config()
    data_path = config['dataset']['data_path']
    task = config['dataset']['task']
    
    if config['dataset']['task'] == 'classification':
        config['dataset']['target'] = target_list
        smiles_data, labels_list = read_smiles_muti(data_path, target_list, task)
        for index in tqdm(range(14324, len(smiles_data)), desc = config['task_name']):
            try:
                converter_data_to_tensor(smiles_data[index], labels_list[index], config)
            except Exception as e:
                print ('Timeout Error Catched!')
                print (e)
                continue

    elif config['dataset']['task'] == 'regression':
        config['dataset']['target'] = target_list
        smiles_data, labels_list = read_smiles_muti(data_path, target_list, task)
        for index in tqdm(range(14324, len(smiles_data)), desc = config['task_name']):
            try:
                converter_data_to_tensor(smiles_data[index], labels_list[index], config)
            except Exception as e:
                print ('Timeout Error Catched!')
                print (e)
                continue
    # HIV：14324
    # SIDER：100