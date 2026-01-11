import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel


def contrastive_loss(z_graph, z_text, temp=0.07):
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)
    
    sim = torch.matmul(z_graph, z_text.T) / temp
    target = torch.arange(z_graph.size(0), device=z_graph.device)
    
    return F.cross_entropy(sim, target)

def contrastive_loss_sampling(z_graph, z_text, batch_idx, train_caption_tensor, batch_size=256, temp=0.07):
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)
    
    device = z_graph.device
    current_batch_size = z_graph.size(0)
    total_samples = train_caption_tensor.size(0)
    
    all_indices = torch.arange(total_samples, device=device)
    mask = torch.ones(total_samples, dtype=torch.bool, device=device)
    mask[batch_idx] = False
    valid_indices = all_indices[mask]
    
    sampled_indices = torch.randint(
        0, 
        len(valid_indices), 
        (current_batch_size, batch_size),
        device=device
    )
    sampled_indices = valid_indices[sampled_indices]
    
    z_text_negatives = train_caption_tensor[sampled_indices]
    z_text_negatives = F.normalize(z_text_negatives, p=2, dim=2)
    pos_sim = torch.sum(z_graph * z_text, dim=1) / temp
    
    neg_sim = torch.bmm(
        z_graph.unsqueeze(1),  
        z_text_negatives.transpose(1, 2)  
    ).squeeze(1) / temp  

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  
    labels = torch.zeros(current_batch_size, dtype=torch.long, device=device)  
    
    loss = F.cross_entropy(logits, labels)
    
    return loss


import torch
import torch.nn.functional as F

def contrastive_loss_softmax_plus_hardneg(z_graph, z_text, batch_idx, train_caption_tensor, 
                                          batch_size=2048, temp=0.07, 
                                          start_hard_rank=5, end_hard_rank=64,
                                          margin=0.2, hard_weight=1.0):
    
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)
    
    device = z_graph.device
    current_batch_size = z_graph.size(0)
    total_samples = train_caption_tensor.size(0)
    
    all_indices = torch.arange(total_samples, device=device)
    mask = torch.ones(total_samples, dtype=torch.bool, device=device)
    mask[batch_idx] = False
    valid_indices = all_indices[mask]
    
    sampled_indices = torch.randint(
        0, 
        len(valid_indices), 
        (current_batch_size, batch_size),
        device=device
    )
    sampled_indices = valid_indices[sampled_indices]
    
    z_text_negatives = train_caption_tensor[sampled_indices]
    z_text_negatives = F.normalize(z_text_negatives, p=2, dim=2)
    
    pos_sim = torch.sum(z_graph * z_text, dim=1)
    
    neg_sim = torch.bmm(
        z_graph.unsqueeze(1),  
        z_text_negatives.transpose(1, 2)  
    ).squeeze(1)

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temp
    labels = torch.zeros(current_batch_size, dtype=torch.long, device=device)
    loss_nce = F.cross_entropy(logits, labels)
    
    k_target = min(end_hard_rank, neg_sim.size(1))
    top_neg_vals, _ = torch.topk(neg_sim, k=k_target, dim=1)
    
    if start_hard_rank < k_target:
        selected_hard_neg = top_neg_vals[:, start_hard_rank:]
    else:
        selected_hard_neg = top_neg_vals[:, -1:]
        
    loss_hard = F.relu(selected_hard_neg - pos_sim.unsqueeze(1) + margin).mean()
    
    total_loss = loss_nce + (hard_weight * loss_hard)
    
    return total_loss


def siglip_loss_sampling(z_graph, z_text, batch_idx, train_caption_tensor, batch_size=256, temp=10.0, bias=-10.0):
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)
    
    device = z_graph.device
    current_batch_size = z_graph.size(0)
    total_samples = train_caption_tensor.size(0)
    
    all_indices = torch.arange(total_samples, device=device)
    mask = torch.ones(total_samples, dtype=torch.bool, device=device)
    mask[batch_idx] = False
    valid_indices = all_indices[mask]
    
    sampled_indices = torch.randint(
        0, 
        len(valid_indices), 
        (current_batch_size, batch_size),
        device=device
    )
    sampled_indices = valid_indices[sampled_indices]
    
    z_text_negatives = train_caption_tensor[sampled_indices]
    z_text_negatives = F.normalize(z_text_negatives, p=2, dim=2)
    
    logits_pos = torch.sum(z_graph * z_text, dim=1) 
    logits_pos = logits_pos * temp + bias
    
    logits_neg = torch.bmm(
        z_graph.unsqueeze(1),  
        z_text_negatives.transpose(1, 2)  
    ).squeeze(1)
    logits_neg = logits_neg * temp + bias
    
    loss_pos = -F.logsigmoid(logits_pos).sum()
    loss_neg = -F.logsigmoid(-logits_neg).sum()
    
    loss = (loss_pos + loss_neg) / current_batch_size
    
    return loss


def compute_kl_losses(z_graph, z_text_batch, temp):
    z_graph_norm = F.normalize(z_graph, p=2, dim=1)
    z_text_norm = F.normalize(z_text_batch, p=2, dim=1)
    
    sim_tt = torch.matmul(z_text_norm, z_text_norm.T) * temp
    target_dist = F.softmax(sim_tt, dim=1) 
    
    sim_mm = torch.matmul(z_graph_norm, z_graph_norm.T) * temp
    log_prob_mm = F.log_softmax(sim_mm, dim=1) 
    
    sim_mt = torch.matmul(z_graph_norm, z_text_norm.T) * temp
    log_prob_mt = F.log_softmax(sim_mt, dim=1) 
    
    loss_u2u = F.kl_div(log_prob_mm, target_dist, reduction='batchmean')
    loss_u2c = F.kl_div(log_prob_mt, target_dist, reduction='batchmean')
    
    return loss_u2u, loss_u2c

def compute_triplet_loss(z_graph, z_text, margin=0.2):
    z_graph_norm = F.normalize(z_graph, p=2, dim=1)
    z_text_norm = F.normalize(z_text, p=2, dim=1)

    scores = torch.matmul(z_graph_norm, z_text_norm.T)
    
    pos_scores = torch.diag(scores)
    
    mask = torch.eye(z_graph.size(0), device=z_graph.device).bool()
    scores_masked = scores.clone()
    scores_masked.masked_fill_(mask, -float('inf'))
    
    neg_scores = scores_masked.max(dim=1)[0]
    
    loss = torch.clamp(neg_scores - pos_scores + margin, min=0).mean()
    return loss


import torch
import torch.nn.functional as F

def hybrid_siglip_range_loss(z_graph, z_text, batch_idx, train_caption_tensor, 
                             batch_size=256, start_hard_rank=5, end_hard_rank=32, 
                             siglip_temp=10.0, siglip_bias=-10.0, 
                             margin=0.2, hard_weight=1.0):
    
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)
    
    device = z_graph.device
    current_batch_size = z_graph.size(0)
    total_samples = train_caption_tensor.size(0)
    
    all_indices = torch.arange(total_samples, device=device)
    mask = torch.ones(total_samples, dtype=torch.bool, device=device)
    mask[batch_idx] = False
    valid_indices = all_indices[mask]
    
    sampled_indices = torch.randint(
        0, 
        len(valid_indices), 
        (current_batch_size, batch_size),
        device=device
    )
    sampled_indices = valid_indices[sampled_indices]
    
    z_text_negatives = train_caption_tensor[sampled_indices]
    z_text_negatives = F.normalize(z_text_negatives, p=2, dim=2)
    
    sim_pos = torch.sum(z_graph * z_text, dim=1)
    
    sim_neg = torch.bmm(
        z_graph.unsqueeze(1),
        z_text_negatives.transpose(1, 2)
    ).squeeze(1)
    
    logits_pos = sim_pos * siglip_temp + siglip_bias
    logits_neg = sim_neg * siglip_temp + siglip_bias
    
    loss_pos = -F.logsigmoid(logits_pos).sum()
    loss_neg = -F.logsigmoid(-logits_neg).sum()
    l_siglip = (loss_pos + loss_neg) / current_batch_size
    
    k_target = min(end_hard_rank, sim_neg.size(1))
    
    top_neg_vals, _ = torch.topk(sim_neg, k=k_target, dim=1)
    
    if start_hard_rank < k_target:
        selected_hard_neg = top_neg_vals[:, start_hard_rank:]
    else:
        selected_hard_neg = top_neg_vals[:, -1:] 
    
    l_hard = F.relu(selected_hard_neg - sim_pos.unsqueeze(1) + margin)
    l_hard = l_hard.mean()
    
    total_loss = l_siglip + (hard_weight * l_hard)
    
    return total_loss

def contrastive_loss_bidirectional(z_graph, z_text, temp=0.07):
    """
    Contrastive loss dans les deux sens (graph->text et text->graph)
    """
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)

    sim = torch.matmul(z_graph, z_text.T) / temp
    
    target = torch.arange(z_graph.size(0), device=z_graph.device)
    
    loss_g2t = F.cross_entropy(sim, target)      
    loss_t2g = F.cross_entropy(sim.T, target)    

    return (loss_g2t + loss_t2g) / 2 

import tqdm
def generate_emb(model,data):
    descriptions = [] 
    for graph in tqdm(data, desc="Extraction des descriptions", total=len(data)):
        descriptions.append(getattr(graph, 'descriptions', '')) 

    embeddings_array = model.encode(
        descriptions, 
        batch_size=32,
        max_length=512, 
    )

    return embeddings_array


import torch
import torch.nn.functional as F


def retrieve_captioning(batch_z_graph, batch_text_emb):
    batch_z_graph = F.normalize(batch_z_graph, p=2, dim=1)
    batch_text_emb = F.normalize(batch_text_emb, p=2, dim=1)
    similarities = batch_z_graph @ batch_text_emb.T
    text_id = similarities.argmax(dim=-1)
    return text_id #(batch_size,)

def retrieve_captioning_topk(batch_z_graph, batch_text_emb, top_k=50):
    batch_z_graph = F.normalize(batch_z_graph, p=2, dim=1)
    batch_text_emb = F.normalize(batch_text_emb, p=2, dim=1)
    similarities = batch_z_graph @ batch_text_emb.T
    topk_text_id = similarities.topk(k=top_k, dim=-1).indices
    return topk_text_id #(batch_size, top_k)

class MolecularCaptionEvaluator:
    BERT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading ChemBERTa on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(self.BERT_MODEL_NAME).to(self.device)
        self.bert_model.eval()
        self.smoothing_func = SmoothingFunction().method1
    
    def compute_bleu4_f1(self, pred: str, ref: str) -> float:
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        weights = (0.25, 0.25, 0.25, 0.25)
        
        bleu_p = sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=self.smoothing_func)
        
        bleu_r = sentence_bleu([pred_tokens], ref_tokens, weights=weights, smoothing_function=self.smoothing_func)
        
        if (bleu_p + bleu_r) == 0:
            return 0.0
            
        return 2 * (bleu_p * bleu_r) / (bleu_p + bleu_r)
    
    
    @torch.no_grad()
    def get_token_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.squeeze(0) 

    
    def compute_bertscore(self, pred: str, ref: str) -> dict:
        pred_emb = self.get_token_embeddings(pred) 
        ref_emb = self.get_token_embeddings(ref) 
        
        pred_emb_norm = F.normalize(pred_emb, p=2, dim=1)
        ref_emb_norm = F.normalize(ref_emb, p=2, dim=1)
        
        sim_matrix = torch.mm(pred_emb_norm, ref_emb_norm.t()) 
        
        precision = sim_matrix.max(dim=1)[0].mean().item()
        
        recall = sim_matrix.max(dim=0)[0].mean().item()
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    
    def evaluate_batch(self, predicted_captions: list[str], reference_captions: list[str]) -> dict:
        bleu_scores = []
        bert_precisions = []
        bert_recalls = []
        bert_f1s = []
        
        for pred, ref in zip(predicted_captions, reference_captions):
            bleu_f1 = self.compute_bleu4_f1(pred, ref)
            bleu_scores.append(bleu_f1)
            
            bert_scores = self.compute_bertscore(pred, ref)
            bert_precisions.append(bert_scores['precision'])
            bert_recalls.append(bert_scores['recall'])
            bert_f1s.append(bert_scores['f1'])
            
        bleu_f1_mean = np.mean(bleu_scores)
        bert_f1_mean = np.mean(bert_f1s)
        
        return {
            'bleu4_f1_mean': bleu_f1_mean,
            'bertscore_precision_mean': np.mean(bert_precisions),
            'bertscore_recall_mean': np.mean(bert_recalls),
            'bertscore_f1_mean': bert_f1_mean,
            'composite_score': (bleu_f1_mean + bert_f1_mean) / 2
        }



if __name__ == "__main__":
    evaluator = MolecularCaptionEvaluator()
    predictions = [
        "This molecule contains benzene rings and hydroxyl groups",
        "The compound has aromatic structure with nitrogen atoms"
    ]
    
    references = [
        "This molecule has benzene rings and hydroxyl functional groups",
        "The compound features an aromatic structure containing nitrogen"
    ]
    
    scores = evaluator.evaluate_batch(predictions, references)
    
    print("Evaluation Results:")
    print(f"BLEU-4 F1: {scores['bleu4_f1_mean']:.4f}")
    print(f"BERTScore Precision: {scores['bertscore_precision_mean']:.4f}")
    print(f"BERTScore Recall: {scores['bertscore_recall_mean']:.4f}")
    print(f"BERTScore F1: {scores['bertscore_f1_mean']:.4f}")
    print(f"Composite Score: {scores['composite_score']:.4f}")