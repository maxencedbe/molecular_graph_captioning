import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


class GINConvWithEdge(MessagePassing):
    """
    GIN layer avec support des edge features (équation 3 du paper)
    z_i^(k+1) = MLP_atom^(k+1) ( z_i^(k) + sum_{j in N(i)} [ z_j^(k) + MLP_bond^(k+1)(e_ij) ] )
    """
    def __init__(self, hidden_dim, edge_dim):
        super(GINConvWithEdge, self).__init__(aggr='add')  # Agrégation par somme
        
        # MLP pour les atomes
        self.mlp_atom = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # MLP pour les bonds (edges)
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
        # Transformer les edge features
        edge_embedding = self.mlp_bond(edge_attr)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_embedding=edge_embedding)
        
        # Appliquer MLP_atom sur (x + aggregation)
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
    def __init__(self, node_feat_dim=119, edge_feat_dim=4, hidden_dim=300, 
                 output_dim=768, max_length=512, num_layers=5):
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
        self.max_length = max_length
        
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.gin_layers = nn.ModuleList([
            GINConvWithEdge(hidden_dim, edge_feat_dim)
            for _ in range(num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_layers)
        ])
        
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
        

        z = self.node_embedding(x)  # [num_nodes, hidden_dim]
        
        for gin_layer, bn in zip(self.gin_layers, self.batch_norms):
            z = gin_layer(z, edge_index, edge_attr)
            z = bn(z)
            z = F.relu(z)
        
        from torch_geometric.utils import to_dense_batch
        z_dense, mask = to_dense_batch(z, batch)  # [B, N, hidden_dim]
        
        z_dense = self.output_projection(z_dense)  # [B, N, output_dim]
        
        batch_size, num_nodes, dim = z_dense.shape
        
        if num_nodes > self.max_length:
            z_dense = z_dense[:, :self.max_length, :]
            mask = mask[:, :self.max_length]
        elif num_nodes < self.max_length:
            padding = torch.zeros(
                batch_size, 
                self.max_length - num_nodes, 
                dim,
                device=z_dense.device
            )
            z_dense = torch.cat([z_dense, padding], dim=1)
            
            mask_padding = torch.zeros(
                batch_size, 
                self.max_length - num_nodes,
                dtype=mask.dtype,
                device=mask.device
            )
            mask = torch.cat([mask, mask_padding], dim=1)
        
        return z_dense, mask

from transformers import T5ForConditionalGeneration, AutoTokenizer

class MoLCABackbone_T5(nn.Module):
    def __init__(self, model_name="GT4SD/multitask-text-and-chemistry-t5-base-standard", 
                 graph_hidden_dim=300, freeze_encoder=True, freeze_decoder=True):
        """
        Architecture avec T5:
        - Encodeur T5: encode les SMILES en texte
        - Décodeur T5: génère des descriptions à partir du graphe + SMILES encodé
        
        Args:
            model_name: modèle T5 pré-entraîné sur chimie
            graph_hidden_dim: dimension de sortie du GNN
            freeze_encoder: geler l'encodeur T5
            freeze_decoder: geler le décodeur T5
        """
        super(MoLCABackbone_T5, self).__init__()
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        self.hidden_size = self.t5_model.config.d_model  # 768 pour T5-base
        
        self.gnn_encoder = GraphT5_GINEncoder(
            node_feat_dim=177,
            edge_feat_dim=30,
            hidden_dim=graph_hidden_dim,
            output_dim=self.hidden_size,
            max_length=128,
            num_layers=5
        )
        
        self.graph_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        if freeze_encoder:
            for param in self.t5_model.encoder.parameters():
                param.requires_grad = False
            print("T5 Encoder frozen")
        
        if freeze_decoder:
            for param in self.t5_model.decoder.parameters():
                param.requires_grad = False
            print("T5 Decoder frozen")
    
    def forward(self, mol_data, smiles_text, prompt="Describe the following molecule:", labels=None, return_loss=True):

        batch_size = mol_data.batch.max().item() + 1
        
        H_graph, graph_mask = self.gnn_encoder(mol_data)  # [B, max_nodes, hidden_size]
        H_graph = self.graph_projection(H_graph)
        
        if isinstance(smiles_text, str):
            smiles_text = [smiles_text] * batch_size
        
        smiles_inputs = self.tokenizer(
            smiles_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(H_graph.device)
        
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(H_graph.device)

        smiles_encoder_outputs = self.t5_model.encoder(
            input_ids=smiles_inputs['input_ids'],
            attention_mask=smiles_inputs['attention_mask']
        )
        H_smiles = smiles_encoder_outputs.last_hidden_state  # [B, smiles_len, hidden_size]
        
        combined_encoder_hidden = torch.cat([H_graph, H_smiles], dim=1)
        combined_attention_mask = torch.cat([
            graph_mask.float(),
            smiles_inputs['attention_mask'].float()
        ], dim=1)
        
        if return_loss:
            if labels is None:
                raise ValueError("labels must be provided when return_loss=True")
            
            decoder_input_ids = labels['input_ids'].to(combined_encoder_hidden.device)
            prompt_input_ids = prompt_inputs['input_ids'].to(combined_encoder_hidden.device) 

            prompt_input_ids_batch = prompt_input_ids.repeat(batch_size, 1)
            decoder_input_ids_concat = torch.cat([prompt_input_ids_batch, decoder_input_ids], dim=1)

            prompt_attention_mask_batch = prompt_inputs['attention_mask'].to(combined_encoder_hidden.device).repeat(batch_size, 1)
            decoder_attention_mask = labels['attention_mask'].to(combined_encoder_hidden.device)
            decoder_attention_mask_concat = torch.cat([prompt_attention_mask_batch, decoder_attention_mask], dim=1)

            prompt_loss_mask = torch.full_like(prompt_input_ids_batch, -100)
            labels_concat = torch.cat([prompt_loss_mask, decoder_input_ids], dim=1)



            outputs = self.t5_model(
                encoder_outputs=(combined_encoder_hidden,), 
                attention_mask=combined_attention_mask,
                decoder_input_ids=decoder_input_ids_concat, # L'input complet
                decoder_attention_mask=decoder_attention_mask_concat, # Le masque complet
                labels=labels_concat # La cible avec masquage de perte
            )
            
            return outputs.loss
        
        else:
            from transformers.modeling_outputs import BaseModelOutput

            decoder_input_ids = prompt_inputs['input_ids'].to(combined_encoder_hidden.device)
            decoder_attention_mask = prompt_inputs['attention_mask'].to(combined_encoder_hidden.device)

            generated_ids = self.t5_model.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=combined_encoder_hidden
                ),
                attention_mask=combined_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                max_length=150,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            generated_text = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            return generated_text
    
    def generate(self, mol_data, smiles_text, max_length=150, num_beams=5):

        self.eval()
        with torch.no_grad():
            return self.forward(mol_data, smiles_text, labels=None, return_loss=False)
