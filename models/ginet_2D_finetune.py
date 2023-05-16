import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


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

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet_2D(nn.Module):
    def __init__(self, task='classification', num_layer=5, embed_dim=256, dropout=0, pooling='mean'):
        super(GINet_2D, self).__init__()
        self.task = task
        self.num_layer = num_layer
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.x_embedding1 = nn.Embedding(num_atom_type, self.embed_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, self.embed_dim)
        self.x_embedding3 = nn.Embedding(num_formal_charge, self.embed_dim)
        self.x_embedding4 = nn.Embedding(num_degree_type, self.embed_dim)
        self.x_embedding5 = nn.Embedding(num_Hs, self.embed_dim)
        self.x_embedding6 = nn.Embedding(num_HybridizationType, self.embed_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        nn.init.xavier_uniform_(self.x_embedding6.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(embed_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(embed_dim))

        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError('Pooling operation not defined!')
        
        # projection head
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True), # first layer
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True), # second layer
            nn.Linear(embed_dim, embed_dim, bias=False), 
            nn.BatchNorm1d(embed_dim)
        )
    
        # fine-tune prediction layers
        if self.task == 'classification':
            self.output_layers = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//2),
                nn.Softplus(),
                nn.Linear(embed_dim//2, 2)
            )
        elif self.task == 'regression':
            self.output_layers = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//2),
                nn.Softplus(),
                nn.Linear(embed_dim//2, 1)
            )
        else:
            raise ValueError('Undefined task type!')

    def forward(self, data):
        h = self.x_embedding1(data.x[:,0]) + self.x_embedding2(data.x[:,1]) + self.x_embedding3(data.x[:,2]) + self.x_embedding4(data.x[:,3]) +self.x_embedding5(data.x[:,4]) +self.x_embedding6(data.x[:,5]) # 分别嵌入节点特征和手型特征(6维节点特征)

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, data.edge_index, data.edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

        if self.pool == None:
            h = h[data.pool_mask]
        else:
            h = self.pool(h, data.batch)
        
        h = self.proj_head(h)

        return self.output_layers(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('NOT LOADED:', name)
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
