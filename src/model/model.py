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
projection_dim = 1024
hiddden_dim_n = 512
hidden_dim_e = 128
import torch.nn as nn

class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim_n=512,hidden_dim_e=hidden_dim_e):
        super(AtomEncoder, self).__init__()
        
        self.feat_dims = [119, 9, 11, 12, 9, 5, 8, 2, 2] 
        self.edge_dims = [22,  6, 2] 
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, hidden_dim) for dim in self.feat_dims
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
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(MessagePassing, self).__init__(aggr='add', flow='source_to_target')

        mlp_in_dim = in_channels + hidden_dim_e 

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



from torch_geometric.nn.conv import MessagePassing
class AttMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.1, num_heads=4, alpha=0.05):
        super(AttMessagePassing, self).__init__(aggr='add', flow='source_to_target')
        
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.head_dim = out_channels // num_heads
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        mlp_in_dim = in_channels + hidden_dim_e

        self.mlp_message = nn.Sequential(
            nn.Linear(mlp_in_dim, out_channels),
            nn.RMSNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )

        self.fc_query = nn.Linear(in_channels, out_channels)
        self.fc_key = nn.Linear(in_channels, out_channels)
        self.att = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.mlp_update = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.RMSNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        self.query = self.fc_query(x).view(-1, self.num_heads, self.head_dim)
        self.key = self.fc_key(x).view(-1, self.num_heads, self.head_dim)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index_i, edge_index_j, size_i):
        query_i = self.query[edge_index_i]
        key_j = self.key[edge_index_j]
        
        att_input = torch.cat([query_i, key_j], dim=-1)
        
        att_scores = self.att(att_input).squeeze(-1)
        att_scores = self.leakyrelu(att_scores)
        
        alpha = torch.zeros_like(att_scores)
        for h in range(self.num_heads):
            alpha[:, h] = softmax(att_scores[:, h], edge_index_i, num_nodes=size_i)
        
        alpha = self.dropout(alpha)
        
        message_input = torch.cat([x_j, edge_attr], dim=-1)
        message = self.mlp_message(message_input)
        
        message = message.view(-1, self.num_heads, self.head_dim)
        message = message * alpha.unsqueeze(-1)
        message = message.view(-1, self.out_channels)
        
        return message

    def update(self, aggr_out, x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(update_input)


class GEncoder(nn.Module):
    def __init__(self, num_layers=3, in_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=0.1):
        super(GEncoder, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.dropout_layer = nn.Dropout(dropout)

        self.input_proj = AtomEncoder()
            
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

        x, edge_attr = self.input_proj(x, edge_attr)
        for conv, norm, ffn in zip(self.layers, self.norms, self.ffn):
            
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            
            
        x_dense, mask = to_dense_batch(x, batch)
        z_graph = self.gap_proj(x_dense) 
        att = self.att_vec(z_graph)
        alpha = self.softmax(att)
        
        h_graph = torch.sum(z_graph * alpha * mask.unsqueeze(-1), dim=1) 
        
        z_graph = self.projection_head(h_graph)

        return z_graph


#from torch_geometric.nn.conv import MessagePassing

class GINConvWithEdge(MessagePassing):
    """
    GIN layer avec support des edge features (équation 3 du paper)
    z_i^(k+1) = MLP_atom^(k+1) ( z_i^(k) + sum_{j in N(i)} [ z_j^(k) + MLP_bond^(k+1)(e_ij) ] )
    """
    def __init__(self, hidden_dim, edge_dim):
        super(GINConvWithEdge, self).__init__(aggr='add')
        
        self.mlp_atom = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.mlp_bond = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: [num_nodes, hidden_dim] - node features
            edge_index: [2, num_edges] - graph connectivity
            edge_attr: [num_edges, edge_dim] - edge features
        """
        edge_embedding = self.mlp_bond(edge_attr)
        
        out = self.propagate(edge_index, x=x, edge_embedding=edge_embedding)
        
        out = self.mlp_atom(x + out)
        
        return out
    
    def message(self, x_j, edge_embedding):
        """
        Crée les messages : z_j^(k) + MLP_bond(e_ij)
        
        Args:
            x_j: [num_edges, hidden_dim] - features des voisins
            edge_embedding: [num_edges, hidden_dim] - edge features transformées
        """
        return x_j + edge_embedding


class GraphT5_GINEncoder(nn.Module):
    """
    Five-layered GIN graph encoder from GraphT5
    """
    def __init__(self, node_feat_dim=177, edge_feat_dim=128, hidden_dim=512, 
                 output_dim=1024, num_layers=6):
        """
        Args:
            node_feat_dim: dimension des features des atomes (ex: 119 pour atom types)
            edge_feat_dim: dimension des features des bonds (ex: 4 pour bond types)
            hidden_dim: dimension cachée du GIN (300 dans GraphMVP)
            output_dim: dimension de sortie (768 pour T5-base, 768 pour GPT-2)
            max_length: longueur max des embeddings de graphe (truncation/padding)
            num_layers: nombre de couches GIN (5 dans le paper)
        """
        super(GraphT5_GINEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.dropout = nn.Dropout(p=0.2)
 
        self.input_proj = AtomEncoder()
        
        self.gin_layers = nn.ModuleList([
            GINConvWithEdge(hidden_dim, edge_feat_dim)
            for _ in range(num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])

        self.gap_proj = nn.Linear(1024, hidden_dim)
        self.projection_head = nn.Linear(output_dim, output_dim)
        self.att_vec = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1) 
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        """
        Args:
            data: PyG Data object avec x, edge_index, edge_attr, batch
        
        Returns:
            G*: [batch_size, max_length, output_dim] - graph embeddings prêts pour cross-attention
            mask: [batch_size, max_length] - masque de padding
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        

        z, edge_attr = self.input_proj(x,edge_attr)  # [num_nodes, hidden_dim]
        
        for gin_layer, bn in zip(self.gin_layers, self.batch_norms):
            z = gin_layer(z, edge_index, edge_attr)
            z = bn(z)
            z = F.relu(z)
            z = self.dropout(z)
        
        z = self.output_projection(z)  # [num_nodes, output_dim]
        
        z, mask = to_dense_batch(z, batch)
        z_graph = self.gap_proj(z) 
        att = self.att_vec(z_graph) 
        alpha = self.softmax(att) 
        h_graph = torch.sum(z * alpha * mask.unsqueeze(-1), dim=1) 
        
        z_graph = self.projection_head(h_graph)     

        return z_graph