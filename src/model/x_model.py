import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel, AutoConfig
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch
# --- Hyperparamètres de Base ---
# Note : Ces valeurs sont indicatives et doivent être ajustées
class HParams:
    GNN_EMB_DIM = 256
    TEXT_MODEL_NAME = 'bert-base-uncased' # Modèle pré-entraîné pour le texte
    EMBEDDING_DIM = 768 # Taille de l'intégration de BERT
    NUM_HEADS = 8
    MARGIN = 0.5
    DROPOUT_RATE = 0.1

H = HParams()

# =================================================================
## 1. Encodeur de Graphe (Molécule)
# =================================================================

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
        
        for conv,_,_ in zip(self.layers, self.norms, self.ffn):
            x = conv(x, edge_index, edge_attr)

        z_graph = self.projection_head(x)

        return z_graph
    


# =================================================================
## 2. Encodeur de Texte (Transformer)
# =================================================================

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        # Utiliser l'encodeur BERT pré-entraîné
        self.bert = AutoModel.from_pretrained(model_name)
        # S'assurer que les dimensions correspondent si elles ont été modifiées
        config = AutoConfig.from_pretrained(model_name)
        self.output_dim = config.hidden_size # 768 pour bert-base-uncased

    def forward(self, input_ids, attention_mask):
        # La sortie (output) est un tuple, le premier élément [0] est l'intégration des jetons
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Sortie : H_text (inclut le jeton [CLS] en position 0)
        # Shape: (Batch_Size * Longueur_Seq, EMBEDDING_DIM)
        return output[0] # token_embeddings

# =================================================================
## 3. Modèle de Cross-Attention (MoLCA Backbone)
# =================================================================

class MoLCABackbone(nn.Module):
    def __init__(self, gnn_input_dim=177):
        super(MoLCABackbone, self).__init__()
        
        gnn_output_dim = H.EMBEDDING_DIM 

        self.gnn_encoder = GEncoder()
        self.text_encoder = TextEncoder(H.TEXT_MODEL_NAME)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=H.EMBEDDING_DIM,
            num_heads=H.NUM_HEADS,
            dropout=H.DROPOUT_RATE,
            batch_first=True
        )
        self.norm = nn.LayerNorm(H.EMBEDDING_DIM)
        

        self.score_head = nn.Sequential(
            nn.Linear(H.EMBEDDING_DIM, H.EMBEDDING_DIM // 2),
            nn.ReLU(),
            nn.Dropout(H.DROPOUT_RATE),
            nn.Linear(H.EMBEDDING_DIM // 2, 1) # Score final
        )

    def forward(self, mol_data, text_data):

        H_mol = self.gnn_encoder(mol_data)
        H_mol, _ = to_dense_batch(H_mol, mol_data.batch)

        H_text = self.text_encoder(text_data['input_ids'], text_data['attention_mask']).squeeze(0)
        
        H_prime_text, attn_weights = self.cross_attention(
            query=H_text,
            key=H_mol,
            value=H_mol
        )
        
        # residual connection 
        H_prime_text = self.norm(H_prime_text + H_text) 
        
        # cls token
        E_paire = H_prime_text[:, 0, :] 
        score = self.score_head(E_paire)
        
        return torch.sigmoid(score.squeeze(1)) 

