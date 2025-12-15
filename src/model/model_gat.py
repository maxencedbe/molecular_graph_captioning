import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from FlagEmbedding import FlagModel


class GEncParams:
    node_feat_dim = 177
    edge_feat_dim = 30

    projection_dim = 1024
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
        
        self.atom_mixer = nn.Sequential(
            nn.Linear(hidden_dim_n, hidden_dim_n * 2),
            nn.LayerNorm(hidden_dim_n * 2),
            nn.GELU(),
            nn.Linear(hidden_dim_n * 2, hidden_dim_n),
            nn.LayerNorm(hidden_dim_n)
        )

        self.edge_embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim_e) for dim in self.edge_dims
        ])

        self.edge_mixer = nn.Sequential(
            nn.Linear(hidden_dim_e, hidden_dim_e * 2),
            nn.LayerNorm(hidden_dim_e * 2),
            nn.GELU(),
            nn.Linear(hidden_dim_e * 2, hidden_dim_e),
            nn.LayerNorm(hidden_dim_e)
        )

    def forward(self, x, edge_attr):
        x_embedding = 0
        for i in range(x.size(1)):
            x_embedding += self.atom_embeddings[i](x[:, i])
        
        x_embedding = self.atom_mixer(x_embedding)
        edge_embedding = 0
        for i in range(edge_attr.size(1)):
            edge_embedding += self.edge_embeddings[i](edge_attr[:, i])
            
        edge_embedding = self.edge_mixer(edge_embedding)

        return x_embedding, edge_embedding


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

        x, edge_attr = self.input_proj(x, edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x_dense, mask = to_dense_batch(x, batch)
        
        z_graph_gap = self.gap_proj(x_dense) 
        att = self.att_vec(z_graph_gap).masked_fill(~mask.unsqueeze(-1), float('-inf'))
        alpha = self.softmax(att)
        h_graph = torch.sum(z_graph_gap * alpha, dim=1) 
        z_graph_pool = self.projection_head(h_graph)

        return z_graph_pool, x_dense, mask
    

class DualEncoder(nn.Module):
    def __init__(self, params=params):
        super(DualEncoder, self).__init__()
        self.device = params.device
        self.text_enc = FlagModel(
                "BAAI/bge-large-en-v1.5", 
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True, 
                device=str(params.device))
        
        self.graph_enc = GEncoder()

    def forward(self, data, descriptions):
        text_emb = self.text_enc.encode(descriptions)
        text_emb = torch.from_numpy(text_emb).float().to(self.device)
        graph_emb,_,_ = self.graph_enc(data)

        return graph_emb,text_emb



'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

class GINEBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(GINEBlock, self).__init__()
        
        # Le cœur du GIN est un MLP qui transforme la SOMME des voisins
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2), # BatchNorm est souvent préféré pour GIN
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # GINEConv : GIN qui accepte des edge_attr
        self.conv = GINEConv(self.mlp, train_eps=True) # train_eps=True permet au modèle d'apprendre le poids du nœud central
        
        self.norm = nn.BatchNorm1d(hidden_dim) # GIN marche souvent mieux avec BatchNorm que RMSNorm
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x_in = x
        
        # GINEConv fait : MLP( (1+eps)*x + sum(relu(edge_attr + x_neighbors)) )
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        return x + x_in # Connexion résiduelle simple

class GEncoder(nn.Module):
    def __init__(self, num_layers=5, params=params, dropout=0.1):
        super(GEncoder, self).__init__()
        
        self.input_proj = MolEncoder(params=params)
        
        # Il faut que les dimensions des edges matchent hidden_dim pour GINE
        # Si MolEncoder sort des edges de taille différente, on ajoute une projection
        self.edge_proj = nn.Linear(params.hidden_dim_e, params.hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GINEBlock(hidden_dim=params.hidden_dim, dropout=dropout)
            )

        # Pooling (On garde ton pooling Attention, il est très bien)
        self.gap_proj = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.gap_act = nn.Tanh()
        self.att_vec = nn.Linear(params.hidden_dim, 1)

        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(params.hidden_dim, params.projection_dim),
            nn.LayerNorm(params.projection_dim), # LayerNorm ici pour le contraste final
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(params.projection_dim, params.projection_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x, edge_attr = self.input_proj(x, edge_attr)
        
        # Projection des edges pour qu'ils aient la même taille que X (requis pour GINE)
        edge_attr = self.edge_proj(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # --- Pooling Identique à avant ---
        x_dense, mask = to_dense_batch(x, batch)
        h_att = self.gap_act(self.gap_proj(x_dense))
        att_scores = self.att_vec(h_att)
        att_scores = att_scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        alpha = F.softmax(att_scores, dim=1)
        h_graph = torch.sum(x_dense * alpha, dim=1) 
        
        z_graph_pool = self.projection_head(h_graph)

        return z_graph_pool, x_dense, mask
        '''