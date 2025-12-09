import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F



class GEncParams:
    node_feat_dim = 177
    edge_feat_dim = 30

    projection_dim = 768

    hidden_dim = 512
    hiddden_dim_n = 512
    hidden_dim_e = 128

params = GEncParams()

class MolEncoder(nn.Module):
    def __init__(self, params=params):
        super(MolEncoder, self).__init__()
        
        hidden_dim_n = params.hiddden_dim_n
        hidden_dim_e = params.hidden_dim_e

        self.feat_dims = [119, 9, 11, 12, 9, 5, 8, 2, 2] 
        self.edge_dims = [22,  6, 2] 
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim_n) for dim in self.feat_dims
        ])

        self.embeddings_e = nn.ModuleList([
            nn.Embedding(dim, hidden_dim_e) for dim in self.feat_dims
        ])

    def forward(self, x, edge_attr):
        x_embedding = 0
        edge_emb = 0
        for i in range(x.size(1)):
            x_embedding += self.embeddings[i](x[:, i])
        
        for i in range(edge_attr.size(1)):
            edge_emb += self.embeddings_e[i](edge_attr[:, i])

        
        return x_embedding, edge_emb



class MessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, params=params, dropout=0.1):
        super(MessagePassing, self).__init__(aggr='add', flow='source_to_target')

        mlp_in_dim = in_channels + params.hidden_dim_e 

        self.mlp_message = nn.Sequential(
            nn.Linear(mlp_in_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.mlp_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp_message(message_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(update_input)



class GEncoder(nn.Module):
    def __init__(self, num_layers=3, hidden_dim=params.hidden_dim, params=params, dropout=0.1):
        super(GEncoder, self).__init__()
        
        hidden_dim = params.hidden_dim
        projection_dim = params.projection_dim 

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.dropout_layer = nn.Dropout(dropout)

        self.input_proj = MolEncoder(params=params)
            
        for i in range(num_layers):
            self.layers.append(MessagePassing(hidden_dim, hidden_dim, params=params, dropout=dropout))
            self.norms.append(nn.RMSNorm(hidden_dim))
            self.ffn.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        self.gap_proj = nn.Linear(hidden_dim, hidden_dim)
        self.att_vec = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.RMSNorm(projection_dim),
            nn.ReLU(),
            self.dropout_layer,
            nn.Linear(projection_dim, projection_dim)
        )


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x, edge_attr = self.input_proj(x, edge_attr)
        for conv, norm, ffn in zip(self.layers, self.norms, self.ffn):
            
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            
            
        x_dense, mask = to_dense_batch(x, batch)
        z_graph = self.gap_proj(x_dense) 
        att = self.att_vec(z_graph)
        alpha = self.softmax(att)
        
        h_graph = torch.sum(z_graph * alpha * mask.unsqueeze(-1), dim=1) 
        
        z_graph_pool = self.projection_head(h_graph)

        return z_graph_pool, z_graph, mask

