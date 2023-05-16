import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as HT
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import BondStereo
from rdkit.Chem import AllChem


num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 4
num_formal_charge = 7
num_degree_type = 8
num_Hs = 5
num_HybridizationType = 7

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 4 
num_bond_is_ring = 2
num_bond_is_aromatic = 2
num_bond_is_conjugated = 2



class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        self.edge_embedding3 = nn.Embedding(num_bond_is_ring, emb_dim)
        self.edge_embedding4 = nn.Embedding(num_bond_is_aromatic, emb_dim)
        self.edge_embedding5 = nn.Embedding(num_bond_is_conjugated, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding3.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding4.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding5.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 5)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) + self.edge_embedding3(edge_attr[:,2]) + self.edge_embedding4(edge_attr[:,3]) + self.edge_embedding5(edge_attr[:,4])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr): # 消息传递，拼接边特征和节点特征
        return x_j + edge_attr

    def update(self, aggr_out): # 聚合更新节点特征
        return self.mlp(aggr_out)


class GINet_2D(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, dropout=0, pool='mean'):
        super(GINet_2D, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.dropout = dropout

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = nn.Embedding(num_formal_charge, emb_dim)
        self.x_embedding4 = nn.Embedding(num_degree_type, emb_dim)
        self.x_embedding5 = nn.Embedding(num_Hs, emb_dim)
        self.x_embedding6 = nn.Embedding(num_HybridizationType, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        h = self.x_embedding1(data.x[:,0]) + self.x_embedding2(data.x[:,1]) + self.x_embedding3(data.x[:,2]) + self.x_embedding4(data.x[:,3]) +self.x_embedding5(data.x[:,4]) +self.x_embedding6(data.x[:,5]) # 分别嵌入节点特征和手型特征(6维节点特征)

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, data.edge_index, data.edge_attr) # 边序号和边属性特征
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)# h: [447, 300] [节点数, 特征维数]
        
        h_global = self.pool(h, data.batch) #全局池化 [16, 300]
        h_global = self.feat_lin(h_global) # 特征拉平 [16, 512]
        out_global = self.out_lin(h_global) #输出 [16, 256]

        h_sub = self.pool(h, data.motif_batch)[1:,:] # [46, 300]
        h_sub = self.feat_lin(h_sub) # [46, 512]
        out_sub = self.out_lin(h_sub) # [46, 256]
        
        return h_global, out_global, out_sub


if __name__ == "__main__":
    model = GINEConv(300)
    print(model)