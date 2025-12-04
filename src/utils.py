import torch
import torch.nn.functional as F


def contrastive_loss(z_graph, z_text, temp=0.07):
    z_graph = F.normalize(z_graph, p=2, dim=1)
    z_text = F.normalize(z_text, p=2, dim=1)

    similarity_matrix = torch.matmul(z_graph, z_text.T) / temp

    batch_size = z_graph.size(0)
    target = torch.arange(batch_size, device=z_graph.device)

    loss_graph_to_text = F.cross_entropy(similarity_matrix, target)
    loss_text_to_graph = F.cross_entropy(similarity_matrix.T, target)

    total_loss = (loss_graph_to_text + loss_text_to_graph) / 2

    return total_loss


def retrieve_captioning(batch_z_graph,batch_text_emb):
    batch_z_graph = F.normalize(batch_z_graph, p=2, dim=1)
    batch_text_emb = F.normalize(batch_text_emb, p=2, dim=1)
    similarities = batch_z_graph @ batch_text_emb.T
    text_id = similarities.argmax(dim=-1)
    return text_id #(batch_size,)


import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class MolecularCaptionEvaluator:
    """
    Evaluator for molecular caption generation using BLEU-4 F1 and BERTScore.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.bert_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
        self.bert_model.eval()
        self.smoothing = SmoothingFunction()
    
    def compute_bleu4_f1(self, predicted_caption, reference_caption):
        """
        Compute BLEU-4 F1 score between predicted and reference captions.
        
        Args:
            predicted_caption: str, generated caption
            reference_caption: str, ground truth caption
            
        Returns:
            float: BLEU-4 F1 score
        """
        pred_tokens = predicted_caption.lower().split()
        ref_tokens = reference_caption.lower().split()
        
        # BLEU-4 (precision-oriented)
        bleu_precision = sentence_bleu(
            [ref_tokens], 
            pred_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing.method1
        )
        
        # Reverse BLEU for recall
        bleu_recall = sentence_bleu(
            [pred_tokens], 
            ref_tokens, 
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing.method1
        )
        
        # F1 score
        if bleu_precision + bleu_recall == 0:
            return 0.0
        
        bleu_f1 = 2 * (bleu_precision * bleu_recall) / (bleu_precision + bleu_recall)
        return bleu_f1
    
    def get_bert_embeddings(self, text):
        """
        Get token embeddings from ChemBERTa model.
        
        Args:
            text: str, input text
            
        Returns:
            torch.Tensor: embeddings of shape (seq_len, hidden_dim)
        """
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)
        
        return embeddings
    
    def compute_bertscore(self, predicted_caption, reference_caption):
        """
        Compute BERTScore using ChemBERTa embeddings.
        
        Args:
            predicted_caption: str, generated caption
            reference_caption: str, ground truth caption
            
        Returns:
            dict: {'precision': float, 'recall': float, 'f1': float}
        """
        pred_emb = self.get_bert_embeddings(predicted_caption)  
        ref_emb = self.get_bert_embeddings(reference_caption)   
        
        pred_emb_norm = F.normalize(pred_emb, p=2, dim=1)
        ref_emb_norm = F.normalize(ref_emb, p=2, dim=1)
        
        sim_matrix = torch.mm(pred_emb_norm, ref_emb_norm.t()) 
        

        precision = sim_matrix.max(dim=1)[0].mean().item()
        recall = sim_matrix.max(dim=0)[0].mean().item()
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_batch(self, predicted_captions, reference_captions):
        """
        Evaluate a batch of predictions.
        
        Args:
            predicted_captions: list of str, generated captions
            reference_captions: list of str, ground truth captions
            
        Returns:
            dict: Average scores for the batch
        """
        bleu_scores = []
        bert_precisions = []
        bert_recalls = []
        bert_f1s = []
        
        for pred, ref in zip(predicted_captions, reference_captions):
            # BLEU-4 F1
            bleu_f1 = self.compute_bleu4_f1(pred, ref)
            bleu_scores.append(bleu_f1)
            
            # BERTScore
            bert_scores = self.compute_bertscore(pred, ref)
            bert_precisions.append(bert_scores['precision'])
            bert_recalls.append(bert_scores['recall'])
            bert_f1s.append(bert_scores['f1'])
        
        return {
            'bleu4_f1_mean': np.mean(bleu_scores),
            'bertscore_precision_mean': np.mean(bert_precisions),
            'bertscore_recall_mean': np.mean(bert_recalls),
            'bertscore_f1_mean': np.mean(bert_f1s),
            'composite_score': (np.mean(bleu_scores) + np.mean(bert_f1s)) / 2
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