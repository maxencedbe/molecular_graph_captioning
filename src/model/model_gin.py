import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GINEConv, GlobalAttention, global_mean_pool, global_max_pool

class GEncParams:
    node_feat_dim = 177
    edge_feat_dim = 30
    
    projection_dim = 1024
    hidden_dim = 768       
    num_heads = 8          
    num_layers = 10      
     
    hidden_dim_n = hidden_dim
    hidden_dim_e = hidden_dim     
    
    device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
    )

params = GEncParams()


class MolEncoder(nn.Module):
    def __init__(self, params=params, dropout=0.1):
        super(MolEncoder, self).__init__()
        
        self.feat_dims = [119, 10, 11, 12, 9, 5, 8, 2, 2]
        self.edge_dims = [22, 6, 2]

        self.atom_embeddings = nn.ModuleList([
            nn.Embedding(dim, params.hidden_dim_n) for dim in self.feat_dims
        ])

        edge_embed_dim = 64 
        self.edge_embeddings = nn.ModuleList([
            nn.Embedding(dim, edge_embed_dim) for dim in self.edge_dims
        ])
        
        total_edge_dim = edge_embed_dim * len(self.edge_dims)
        self.edge_proj = nn.Sequential(
            nn.Linear(total_edge_dim, params.hidden_dim_e),
            nn.LayerNorm(params.hidden_dim_e),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.atom_post_proj = nn.Sequential(
            nn.Linear(params.hidden_dim_n, params.hidden_dim),
            nn.LayerNorm(params.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, batch):
        x = sum(emb(batch.x[:, i]) for i, emb in enumerate(self.atom_embeddings))
        x = self.atom_post_proj(x)

        edge_feats = [emb(batch.edge_attr[:, i]) for i, emb in enumerate(self.edge_embeddings)]
        edge_attr = torch.cat(edge_feats, dim=-1) 
        edge_attr = self.edge_proj(edge_attr)

        return x, edge_attr

class GINETransformerBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.conv = GINEConv(gin_mlp, train_eps=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr):
        x_norm = self.norm1(x)
        x_out = self.conv(x_norm, edge_index, edge_attr=edge_attr)
        x = x + x_out
        
        x_norm = self.norm2(x)
        x_out = self.ffn(x_norm)
        x = x + x_out
        
        return x

class VirtualNodeModule(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, batch, vn_embedding):
        gx = global_mean_pool(x, batch)
        
        vn_embedding = vn_embedding + gx
        vn_embedding = self.mlp(vn_embedding)
        
        x = x + vn_embedding[batch]
        return x, vn_embedding

class GEncoder(nn.Module):
    def __init__(self, params=params, dropout=0.1):
        super(GEncoder, self).__init__()
        
        self.input_proj = MolEncoder(params=params, dropout=dropout)
        
        self.virtual_node_embedding = nn.Embedding(1, params.hidden_dim)
        self.virtual_node_updater = VirtualNodeModule(params.hidden_dim, dropout)
        
        self.layers = nn.ModuleList()
        for _ in range(params.num_layers):
            self.layers.append(
                GINETransformerBlock(hidden_dim=params.hidden_dim, dropout=dropout)
            )

        self.pool_gate_nn = nn.Sequential(
            nn.Linear(params.hidden_dim, params.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(params.hidden_dim // 2, 1)
        )
        self.att_pool = GlobalAttention(gate_nn=self.pool_gate_nn)
        self.pooling_norm = nn.LayerNorm(params.hidden_dim * 3)

        self.projection_head = nn.Sequential(
            nn.Linear(params.hidden_dim * 3, params.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(params.hidden_dim, params.projection_dim)  
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.1, 0.1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x, edge_attr = self.input_proj(data)

        batch_size = batch.max().item() + 1
        vn_emb = self.virtual_node_embedding(torch.zeros(batch_size, dtype=torch.long, device=x.device))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x, vn_emb = self.virtual_node_updater(x, batch, vn_emb)

        u_att = self.att_pool(x, batch)
        u_max = global_max_pool(x, batch)
        u_mean = global_mean_pool(x, batch)
        
        z_graph = torch.cat([u_att, u_max, u_mean], dim=1)
        z_graph = self.pooling_norm(z_graph)
        
        z_projected = self.projection_head(z_graph)
        z_projected = F.normalize(z_projected, p=2, dim=1)

        x_dense, mask = to_dense_batch(x, batch)

        return z_projected, x_dense, mask