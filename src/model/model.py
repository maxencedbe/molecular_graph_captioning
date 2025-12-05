import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


node_feat_dim = 177
edge_feat_dim = 30
hidden_dim = 512
projection_dim = 768


class MessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(MessagePassing, self).__init__(aggr='add', flow='source_to_target')

        mlp_in_dim = in_channels + edge_feat_dim 

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
    def __init__(self, num_layers=6, in_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=0.1):
        super(GEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.dropout_layer = nn.Dropout(dropout)

        if in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()
            
        for i in range(num_layers):
            self.layers.append(MessagePassing(hidden_dim, hidden_dim, dropout=dropout))
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

        x = self.input_proj(x)
        
        for conv, norm, ffn in zip(self.layers, self.norms, self.ffn):
            x_res = x 
            
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x_res + x_new)
            
            x_res = x
            x_new = ffn(x)
            x = norm(x_res + self.dropout_layer(x_new))
            
        x_dense, mask = to_dense_batch(x, batch)
        z_graph = self.gap_proj(x_dense) 
        att = self.att_vec(z_graph)
        alpha = self.softmax(att)
        
        h_graph = torch.sum(z_graph * alpha * mask.unsqueeze(-1), dim=1) 
        
        z_graph = self.projection_head(h_graph)

        return z_graph


class GEncoder2(nn.Module):
    def __init__(self, num_layers=3, in_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=0.1):
        super(GEncoder2, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn = nn.ModuleList() 
        self.dropout_layer = nn.Dropout(dropout)
        
        if in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        for i in range(num_layers):
            self.layers.append(AttMessagePassing(hidden_dim, hidden_dim, dropout=dropout))
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
        
        x = self.input_proj(x)
        
        for conv, norm, ffn in zip(self.layers, self.norms, self.ffn):
            x_res = x 
            
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x_res + x_new)
            
            x_res = x
            x_new = ffn(x)
            x = norm(x_res + self.dropout_layer(x_new))

        x_dense, mask = to_dense_batch(x, batch)
        z_graph = self.gap_proj(x_dense) 
        att = self.att_vec(z_graph) 
        alpha = self.softmax(att) 
        h_graph = torch.sum(z_graph * alpha * mask.unsqueeze(-1), dim=1) 
        
        # 4. Projection finale
        z_graph = self.projection_head(h_graph)

        return z_graph