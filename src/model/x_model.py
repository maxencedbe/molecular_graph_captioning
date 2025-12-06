import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoModel, AutoConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_batch


class HParams:
    EMBEDDING_DIM = 768 
    NUM_HEADS = 8
    MARGIN = 0.5
    DROPOUT_RATE = 0.1

H = HParams()



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



class MoLCABackbone(nn.Module):
    def __init__(self, gnn_input_dim=177, model_name="gpt2"):
        super(MoLCABackbone, self).__init__()
        
        # Encodeur de graphe
        self.gnn_encoder = GEncoder()
        
        # LLM pré-chargé (GPT-2)
        self.llm = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Dimension des embeddings GPT-2
        self.hidden_size = self.llm.config.hidden_size
        
        # Projection des embeddings de graphe vers l'espace GPT-2
        self.graph_projection = nn.Sequential(
            nn.Linear(H.EMBEDDING_DIM, self.hidden_size),  # 256 = sortie de GEncoder
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, mol_data, batch_text, return_loss=True):
        """
        Args:
            mol_data: données du graphe moléculaire
            batch_text: dict avec 'input_ids' et 'attention_mask' ou textes bruts
            return_loss: si True, calcule la loss, sinon génère du texte
        
        Returns:
            loss ou texte généré selon le mode
        """
        H_mol = self.gnn_encoder(mol_data)
        H_mol, mask = to_dense_batch(H_mol, mol_data.batch)
        batch_size = H_mol.size(0)
 
        H_mol_projected = self.graph_projection(H_mol)  # [batch, num_nodes, hidden_size]
        
        if return_loss:

            input_ids = batch_text['input_ids']
            attention_mask = batch_text['attention_mask']
            
            text_embeds = self.llm.transformer.wte(input_ids)  # [batch, seq_len, hidden_size]
            
            combined_embeds = torch.cat([H_mol_projected, text_embeds], dim=1)
            
            graph_attention_mask = mask.float()
            combined_attention_mask = torch.cat([
                graph_attention_mask, 
                attention_mask.float()
            ], dim=1)
            
            graph_length = H_mol_projected.size(1)
            labels = torch.cat([
                torch.full((batch_size, graph_length), -100, device=input_ids.device, dtype=input_ids.dtype),
                input_ids
            ], dim=1)
            
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=labels
            )
            
            return outputs.loss
        
        else:

            if isinstance(batch_text, str):
                batch_text = [batch_text] * batch_size
            
            inputs = self.tokenizer(
                batch_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(H_mol.device)
            
            # Obtenir les embeddings du prompt
            prompt_embeds = self.llm.transformer.wte(inputs['input_ids'])
            
            # Concaténer
            combined_embeds = torch.cat([H_mol_projected, prompt_embeds], dim=1)
            
            # Créer le masque d'attention
            graph_attention_mask = mask.float()
            combined_attention_mask = torch.cat([
                graph_attention_mask,
                inputs['attention_mask'].float()
            ], dim=1)
            
            # Générer avec le LLM
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                max_length=combined_embeds.size(1) + 50,  # Ajustable
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            prompt_length = combined_embeds.size(1)
            generated_tokens = outputs[:, prompt_length:]
            generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text
    
    def generate(self, mol_data, prompt="Description: ", max_length=50, temperature=0.7):
        """
        Méthode utilitaire pour générer du texte
        
        Args:
            mol_data: données du graphe moléculaire
            prompt: prompt textuel
            max_length: nombre de tokens à générer
            temperature: température de génération
        
        Returns:
            generated_text: liste de textes générés
        """
        return self.forward(mol_data, prompt, return_loss=False)
