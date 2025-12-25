import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F

class GEncParams:
    node_feat_dim = 177
    edge_feat_dim = 30

    projection_dim = 3840
    num_heads = 8
    hidden_dim = 512
    hidden_dim_n = 512
    hidden_dim_e = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = GEncParams()

class MolEncoder(nn.Module):
    def __init__(self, params=params):
        super(MolEncoder, self).__init__()
        
        hidden_dim_n = params.hidden_dim_n
        hidden_dim_e = params.hidden_dim_e

        self.feat_dims = [119, 10, 11, 12, 9, 5, 8, 2, 2]
        self.edge_dims = [22, 6, 2]

        self.atom_embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim_n) for dim in self.feat_dims
        ])

        self.edge_embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim_e) for dim in self.edge_dims
        ])
        
    def forward(self, batch):
        x = sum(emb(batch.x[:, i]) for i, emb in enumerate(self.atom_embeddings))
        edge_attr = sum(emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_embeddings))

        return x, edge_attr



from torch_geometric.nn import GATv2Conv
class GATv2Block(nn.Module):
    def __init__(self, hidden_dim, edge_dim, heads=4, dropout=0.1):
        super(GATv2Block, self).__init__()
        
        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads, 
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=False 
        )
        
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.RMSNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):

        x_attn = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x + x_attn)
        x_ffn = self.ffn(x)
        x = self.norm2(x + x_ffn)
        
        return x


class GEncoder(nn.Module):
    def __init__(self, num_layers=5, params=params, dropout=0.1):
        super(GEncoder, self).__init__()
        
        self.input_proj = MolEncoder(params=params)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Block(
                    hidden_dim=params.hidden_dim, 
                    edge_dim=params.hidden_dim_e,
                    heads=params.num_heads,
                    dropout=dropout
                )
            )

        self.gap_proj = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.att_vec = nn.Linear(params.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

        self.projection_head = nn.Sequential(
            nn.Linear(params.hidden_dim, params.projection_dim),
            nn.RMSNorm(params.projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(params.projection_dim, params.projection_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x, edge_attr = self.input_proj(data)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x_dense, mask = to_dense_batch(x, batch)
        
        z_graph_gap = self.gap_proj(x_dense) 
        att = self.att_vec(z_graph_gap).masked_fill(~mask.unsqueeze(-1), float('-inf'))
        alpha = self.softmax(att)
        h_graph = torch.sum(z_graph_gap * alpha, dim=1) 
        z_graph_pool = self.projection_head(h_graph)

        return z_graph_pool, x_dense, mask
