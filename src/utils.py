import torch
import torch.nn.functional as F
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModel


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

def retrieve_captioning(batch_z_graph, batch_text_emb):
    batch_z_graph = F.normalize(batch_z_graph, p=2, dim=1)
    batch_text_emb = F.normalize(batch_text_emb, p=2, dim=1)
    similarities = batch_z_graph @ batch_text_emb.T
    text_id = similarities.argmax(dim=-1)
    return text_id #(batch_size,)


class MolecularCaptionEvaluator:
    """
    Évaluateur pour les légendes moléculaires utilisant BLEU-4 F1 et BERTScore 
    (basé sur ChemBERTa).
    """
    BERT_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
    
    def __init__(self, device: str = None):
        """Initialise le modèle ChemBERTa pour BERTScore."""
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading ChemBERTa on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL_NAME)
        self.bert_model = AutoModel.from_pretrained(self.BERT_MODEL_NAME).to(self.device)
        self.bert_model.eval()
        self.smoothing_func = SmoothingFunction().method1 # Precompute smoothing function

    # --------------------------------------------------------
    # I. Calculation of Lexical Metrics (BLEU-4 F1)
    # --------------------------------------------------------
    
    def compute_bleu4_f1(self, pred: str, ref: str) -> float:
        """Calculates BLEU-4 F1 (precision & recall based on n-grams)."""
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        weights = (0.25, 0.25, 0.25, 0.25)
        
        # Précision (BLEU P): Référence vs Prédiction
        bleu_p = sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=self.smoothing_func)
        
        # Rappel (BLEU R): Prédiction vs Référence (technique du Reverse BLEU)
        bleu_r = sentence_bleu([pred_tokens], ref_tokens, weights=weights, smoothing_function=self.smoothing_func)
        
        if (bleu_p + bleu_r) == 0:
            return 0.0
            
        return 2 * (bleu_p * bleu_r) / (bleu_p + bleu_r)

    # --------------------------------------------------------
    # II. Outils d'Encodage pour BERTScore
    # --------------------------------------------------------
    
    @torch.no_grad()
    def get_token_embeddings(self, text: str) -> torch.Tensor:
        """Obtient les embeddings de tokens du modèle ChemBERTa (non [CLS])."""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        outputs = self.bert_model(**inputs)
        # Retourne le dernier état caché (B=1, Seq_len, H)
        # On utilise squeeze(0) pour obtenir (Seq_len, H)
        return outputs.last_hidden_state.squeeze(0) 

    # --------------------------------------------------------
    # III. Calcul des Métriques Sémantiques (BERTScore)
    # --------------------------------------------------------
    
    def compute_bertscore(self, pred: str, ref: str) -> dict:
        """Calcule la Précision, le Rappel et le F1 du BERTScore."""
        
        pred_emb = self.get_token_embeddings(pred) 
        ref_emb = self.get_token_embeddings(ref) 
        
        # Normalisation L2
        pred_emb_norm = F.normalize(pred_emb, p=2, dim=1)
        ref_emb_norm = F.normalize(ref_emb, p=2, dim=1)
        
        # Matrice de Similarité Cosinus (Prédits x Référence.T)
        sim_matrix = torch.mm(pred_emb_norm, ref_emb_norm.t()) 
        
        # 1. Précision BERTScore (Sim_P): Max des colonnes, moyenné sur la Prédiction (dim=1)
        # Pour chaque token Prédit, quel est le meilleur token de Référence?
        precision = sim_matrix.max(dim=1)[0].mean().item()
        
        # 2. Rappel BERTScore (Sim_R): Max des lignes, moyenné sur la Référence (dim=0)
        # Pour chaque token de Référence, quel est le meilleur token Prédit?
        recall = sim_matrix.max(dim=0)[0].mean().item()
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    # --------------------------------------------------------
    # IV. Évaluation du Lot (API d'exécution)
    # --------------------------------------------------------
    
    def evaluate_batch(self, predicted_captions: list[str], reference_captions: list[str]) -> dict:
        """Évalue un lot de prédictions et retourne les scores moyens."""
        
        bleu_scores = []
        bert_precisions = []
        bert_recalls = []
        bert_f1s = []
        
        for pred, ref in zip(predicted_captions, reference_captions):
            # 1. Calcul BLEU
            bleu_f1 = self.compute_bleu4_f1(pred, ref)
            bleu_scores.append(bleu_f1)
            
            # 2. Calcul BERTScore
            bert_scores = self.compute_bertscore(pred, ref)
            bert_precisions.append(bert_scores['precision'])
            bert_recalls.append(bert_scores['recall'])
            bert_f1s.append(bert_scores['f1'])
            
        
        # Calcul des moyennes
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