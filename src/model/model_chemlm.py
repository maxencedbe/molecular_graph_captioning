import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.model_gat import GEncoder, GEncParams

class GraphChemLLM(nn.Module):
    def __init__(self, llm_model_id="AI4Chem/ChemLLM-2b-1_5", gnn_output_dim=512, freeze_llm=True, freeze_gnn=False, freeze_projector=False):
        """
        """
        super(GraphChemLLM, self).__init__()
        

        print(f"Loading LLM: {llm_model_id}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_id, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gnn = GEncoder()
        
        self.gnn_output_dim = gnn_output_dim
        self.llm_hidden_dim = self.llm.config.hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(self.gnn_output_dim, self.llm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim)
        )
        
        if freeze_llm:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
        
        if freeze_projector:
            print("Freezing projector parameters...")
            for param in self.projector.parameters():
                param.requires_grad = False

        if freeze_gnn:
            print("Freezing GNN parameters...")
            for param in self.gnn.parameters():
                param.requires_grad = False

    def forward(self, batch_graph, input_ids, attention_mask=None, labels=None):
        """

        """

        input_embeds_layer = self.llm.get_input_embeddings()
        text_embeds = input_embeds_layer(input_ids) # [Batch, Seq_Len, LLM_Dim]


        z_graph, z_dense, z_mask = self.gnn(batch_graph) 

        graph_emb = self.projector(z_dense)  # [B, Seq_Len_Graph, LLM_Dim]
        

        graph_emb = graph_emb.to(text_embeds.dtype)
        inputs_embeds = torch.cat([graph_emb, text_embeds], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat([z_mask, attention_mask], dim=1)

        if labels is not None:
            graph_mask_labels = torch.full((labels.shape[0], z_dense.shape[1]), -100, device=labels.device, dtype=labels.dtype)
            labels = torch.cat([graph_mask_labels, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

    def generate(self, batch_graph, prompt_text, max_new_tokens=50, temperature=0.7):
        """

        """
        device = batch_graph.x.device 
        
        with torch.no_grad():
            z_graph, _, _ = self.gnn(batch_graph)
            graph_emb = self.projector(z_graph).unsqueeze(1)
            
            target_dtype = self.llm.model.embed_tokens.weight.dtype
            graph_emb = graph_emb.to(target_dtype)

        inputs = self.tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        input_embeds_layer = self.llm.get_input_embeddings()
        text_embeds = input_embeds_layer(input_ids)

        inputs_embeds = torch.cat([graph_emb, text_embeds], dim=1)
        
        graph_mask = torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([graph_mask, attention_mask], dim=1)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )


        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)