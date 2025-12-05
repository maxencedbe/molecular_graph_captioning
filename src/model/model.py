import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import to_dense_batch

node_feat_dim = 177
edge_feat_dim = 30
hidden_dim = 512
projection_dim = 768


class GEncoder(nn.Module):
    def __init__(self, num_layers=6, in_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=0.1):
        super(GEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn = nn.ModuleList() 
        self.dropout = nn.Dropout(dropout)
        
        current_dim = in_dim

        self.layers.append(MessagePassing(current_dim, hidden_dim, dropout=dropout))
        self.norms.append(nn.RMSNorm(hidden_dim))
        current_dim = hidden_dim

        for _ in range(num_layers - 1):
            self.layers.append(MessagePassing(current_dim, hidden_dim, dropout=dropout))
            self.norms.append(nn.RMSNorm(hidden_dim))
            self.ffn.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.RMSNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )   

        self.gap_proj = nn.Linear(hidden_dim,hidden_dim)
        self.att_vec = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for conv, norm, ffn in zip(self.layers, self.norms,self.ffn):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = ffn(x_new)
            x_new = self.dropout(x_new)
            if x.shape[-1] == x_new.shape[-1]:
                x = x + x_new
            else:
                x = x_new

        # h_graph = global_add_pool(x, batch)

        x_dense, mask = to_dense_batch(x, batch)
        z_graph = self.gap_proj(x_dense) #(batch,node,hidden_dim)
        att = self.att_vec(z_graph) #(batch,node,1)
        alpha = self.softmax(att)
        h_graph = torch.sum(z_graph * alpha * mask.unsqueeze(-1), dim=1) #(batch,hidden_dim)
        z_graph = self.projection_head(h_graph)

        return z_graph


class GEncoder2(nn.Module):
    def __init__(self, num_layers=3, in_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=0.1):
        super(GEncoder2, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn = nn.ModuleList() 
        self.dropout = nn.Dropout(dropout)
        
        current_dim = in_dim

        self.layers.append(MessagePassing(current_dim, hidden_dim, dropout=dropout))
        self.norms.append(nn.RMSNorm(hidden_dim))
        current_dim = hidden_dim

        for _ in range(num_layers - 1):
            self.layers.append(MessagePassing(current_dim, hidden_dim, dropout=dropout))
            self.norms.append(nn.RMSNorm(hidden_dim))
            self.ffn.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.RMSNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim)
        )   

        self.gap_proj = nn.Linear(hidden_dim,hidden_dim)
        self.att_vec = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for conv, norm, ffn in zip(self.layers, self.norms,self.ffn):
            x_new = conv(x, edge_index, edge_attr)

            x = x_new

        # h_graph = global_add_pool(x, batch)

        x_dense, mask = to_dense_batch(x, batch)
        z_graph = self.gap_proj(x_dense) #(batch,node,hidden_dim)
        att = self.att_vec(z_graph) #(batch,node,1)
        alpha = self.softmax(att)
        h_graph = torch.sum(z_graph * alpha * mask.unsqueeze(-1), dim=1) #(batch,hidden_dim)
        z_graph = self.projection_head(h_graph)

        return z_graph

class MessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(MessagePassing, self).__init__(aggr='add', flow='source_to_target')

        mlp_in_dim = in_channels + edge_feat_dim

        self.mlp_message = nn.Sequential(
            nn.Linear(mlp_in_dim, out_channels),
            #nn.RMSNorm(out_channels),
            nn.ReLU(),
            #nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )

        self.mlp_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            #nn.RMSNorm(out_channels),
            nn.ReLU(),
            #nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp_message(message_input)

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(update_input)
    




edge_feat_dim = 30


class AttMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.1, num_heads=4, alpha=0.05):
        super(AttMessagePassing, self).__init__(aggr='add', flow='source_to_target')

        self.num_heads = num_heads
        self.out_channels = out_channels
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        mlp_in_dim = in_channels + edge_feat_dim

        # Message MLP
        self.mlp_message = nn.Sequential(
            nn.Linear(mlp_in_dim, out_channels),
            nn.RMSNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )

        # Attention mechanism (GAT-style)
        # Transform node features for attention
        self.fc_query = nn.Linear(in_channels, out_channels)
        self.fc_key = nn.Linear(in_channels, out_channels)
        
        #self.att = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim) * 0.02)
        self.att = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Update MLP
        self.mlp_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.RMSNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # Compute queries and keys for attention
        self.query = self.fc_query(x).view(-1, self.num_heads, self.head_dim)
        self.key = self.fc_key(x).view(-1, self.num_heads, self.head_dim)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index_i, edge_index_j, size_i):
        
        query_i = self.query[edge_index_i]  
        key_j = self.key[edge_index_j]  
        
        num_edges = x_j.size(0)
        att_input = torch.cat([query_i, key_j], dim=-1)  # [num_edges, num_heads, 2 * head_dim]
        
        # Compute attention scores per head
         #att_scores = (att_input * self.att.unsqueeze(0)).sum(dim=-1)  # [num_edges, num_heads]
        att_scores = self.att(att_input).squeeze(-1)  # [num_edges, num_heads]
        att_scores = self.leakyrelu(att_scores)
        
        # Normalize attention scores using softmax (per target node, per head)
        # We need to apply softmax separately for each head
        alpha = torch.zeros_like(att_scores)
        for h in range(self.num_heads):
            alpha[:, h] = softmax(att_scores[:, h], edge_index_i, num_nodes=size_i)
        
        alpha = self.dropout(alpha)
        
        # Compute message with edge features
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        message = self.mlp_message(message_input)  # [num_edges, out_channels]
        
        # Apply attention weights
        message = message.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        message = message * alpha.unsqueeze(-1)  # [num_edges, num_heads, head_dim]
        message = message.view(-1, self.out_channels)  # [num_edges, out_channels]
        
        return message

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(update_input)