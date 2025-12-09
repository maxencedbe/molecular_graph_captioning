import torch
import torch.nn as nn
from src.model.model_Genc import GEncoder, GEncParams
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch_geometric.utils import to_dense_batch

enc_params = GEncParams()
class GraphCapT5(nn.Module):
    def __init__(self, model_name="laituan245/molt5-base", 
                 enc_params=enc_params, freeze_encoder=True, freeze_decoder=True):
        """
        faudra qu'on fasse des type scripts plus tard
        """
        super(GraphCapT5, self).__init__()
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.hidden_size = self.t5_model.config.d_model        
        
        self.gnn_encoder = GEncoder(params=enc_params)
        
        self.graph_projection_pool = nn.Sequential(
            nn.Linear(enc_params.projection_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
        self.graph_projection = nn.Sequential(
            nn.Linear(enc_params.hidden_dim, self.hidden_size),
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
        
        H_graph_pool, H_graph, graph_mask = self.gnn_encoder(mol_data) 
        H_graph_pool = self.graph_projection_pool(H_graph_pool).unsqueeze(1)
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

        prompt_encoder_outputs = self.t5_model.encoder(
            input_ids=prompt_inputs['input_ids'],
            attention_mask=prompt_inputs['attention_mask']
        )
        H_prompt = prompt_encoder_outputs.last_hidden_state  # [1, prompt_len, hidden_size]
        H_prompt = H_prompt.expand(batch_size, -1, -1)  # [B, prompt_len, hidden_size]
        
        graph_pool_mask = torch.ones(batch_size, H_graph_pool.size(1), dtype=torch.float32, device=H_graph.device)
        combined_encoder_hidden = torch.cat([H_graph_pool, H_graph, H_smiles], dim=1)
        combined_attention_mask = torch.cat([
            graph_pool_mask,
            graph_mask.float(),
            smiles_inputs['attention_mask'].float()
        ], dim=1)
        
        if return_loss:
            if labels is None:
                raise ValueError("labels must be provided when return_loss=True")
            
            decoder_input_ids = labels['input_ids'].to(combined_encoder_hidden.device)

            outputs = self.t5_model(
                encoder_outputs=(combined_encoder_hidden,), 
                attention_mask=combined_attention_mask,
                labels=decoder_input_ids 
            )
            
            return outputs.loss
        
        else:
            from transformers.modeling_outputs import BaseModelOutput

            generated_ids = self.t5_model.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=combined_encoder_hidden
                ),
                attention_mask=combined_attention_mask,
                max_length=512,
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
