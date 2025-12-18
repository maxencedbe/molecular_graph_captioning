import torch
import tqdm
import pandas as pd
import os
from torch.utils.data import DataLoader

# Importations depuis ton projet src
# Assure-toi que ce fichier est à la racine ou que le PYTHONPATH est configuré
from src.utils import MolecularCaptionEvaluator, retrieve_captioning 
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn, load_id2emb, embdict_to_tensor
from src.model.model_gat import GEncoder

# --- Configuration ---
BATCH_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Chemins de fichiers (identiques au script d'entraînement)
TRAIN_DATA_FILE = "src/data/train_graphs.pkl"
TRAIN_EMB_CSV = "src/data/train_embeddings_bge.csv"
MODEL_PATH = "src/saved_model/best_model.pth"
OUTPUT_CSV = "src/results/train_scoring_results.csv" # Pour sauvegarder les prédictions

def score_dataset(model, dataloader, retrieval_bank, data_list, evaluator, device):
    """
    Évalue le modèle sur le dataset donné en utilisant le retrieval.
    """
    model.eval()
    
    # Listes pour stocker toutes les phrases pour une évaluation globale (plus précis pour BLEU)
    all_predictions = []
    all_ground_truths = []
    
    # Métriques accumulées pour moyenne par batch (comme dans ton train.py)
    total_score = 0
    total_bleu = 0
    total_bert = 0
    num_samples = 0
    
    progress_bar = tqdm.tqdm(dataloader, desc="Scoring Train Set")
    
    with torch.no_grad():
        for batch_graph, _ in progress_bar:
            batch_graph = batch_graph.to(device)
            current_batch_size = batch_graph.num_graphs
            
            # 1. Encodage des Graphes
            z_graph, _, _ = model(batch_graph)
            
            # 2. Retrieval : Trouver les indices des captions les plus proches dans la banque
            # On cherche dans train_caption_tensor (le retrieval_bank)
            text_ids = retrieve_captioning(z_graph, retrieval_bank)
            
            # 3. Mapping : Indices -> Texte Prédit
            # data_list contient les objets originaux, on suppose que l'ordre des embeddings 
            # correspond à l'ordre de la liste data_list (ce qui est le cas standard)
            pred_captions = [data_list[i].description for i in text_ids.cpu().numpy()]
            
            # 4. Vérité Terrain (Ground Truth)
            individual_graphs = batch_graph.to_data_list()
            true_captions = [graph.description for graph in individual_graphs]
            
            # Stockage
            all_predictions.extend(pred_captions)
            all_ground_truths.extend(true_captions)
            
            # 5. Évaluation du batch (pour logging immédiat)
            batch_scores = evaluator.evaluate_batch(pred_captions, true_captions)
            
            total_score += batch_scores['composite_score'] * current_batch_size
            total_bleu += batch_scores['bleu4_f1_mean'] * current_batch_size
            total_bert += batch_scores['bertscore_f1_mean'] * current_batch_size
            num_samples += current_batch_size
            
            progress_bar.set_postfix({
                "Avg Score": f"{total_score / num_samples:.4f}",
                "BLEU-4": f"{total_bleu / num_samples:.4f}"
            })

    # Calcul des moyennes finales
    avg_score = total_score / num_samples
    avg_bleu = total_bleu / num_samples
    avg_bert = total_bert / num_samples
    
    return {
        "composite_score": avg_score,
        "bleu4": avg_bleu,
        "bertscore": avg_bert,
        "predictions": all_predictions,
        "ground_truths": all_ground_truths
    }

def main():
    print(f"--- Initialisation sur {DEVICE} ---")

    # 1. Chargement des données brutes et embeddings
    print("Chargement des données...")
    train_data_list = load_data(TRAIN_DATA_FILE)
    train_emb_dict = load_id2emb(TRAIN_EMB_CSV)
    
    # Création de la banque de retrieval (Tensor)
    # C'est la base de données dans laquelle on va chercher les voisins
    train_caption_tensor = embdict_to_tensor(train_emb_dict).to(DEVICE)
    
    # Création du Dataset et DataLoader
    train_dataset = PreprocessedGraphDataset(TRAIN_DATA_FILE, train_emb_dict, encode_feat=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # Important: False pour garder l'ordre si on veut analyser après
        collate_fn=collate_fn
    )

    # 2. Chargement du Modèle
    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model = GEncoder().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        # map_location assure que ça charge même si tu passes de GPU à CPU
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Poids du modèle chargés avec succès.")
    else:
        raise FileNotFoundError(f"Le fichier modèle n'a pas été trouvé : {MODEL_PATH}")

    # 3. Initialisation de l'évaluateur
    evaluator = MolecularCaptionEvaluator(device=DEVICE)

    # 4. Lancement du Scoring
    print("Démarrage de l'évaluation sur le Train Set...")
    results = score_dataset(
        model=model,
        dataloader=train_loader,
        retrieval_bank=train_caption_tensor,
        data_list=train_data_list,
        evaluator=evaluator,
        device=DEVICE
    )

    # 5. Affichage et Sauvegarde des Résultats
    print("\n" + "="*30)
    print("RÉSULTATS FINAUX SUR LE TRAIN SET")
    print("="*30)
    print(f"Composite Score : {results['composite_score']:.4f}")
    print(f"BLEU-4 F1       : {results['bleu4']:.4f}")
    print(f"BERTScore F1    : {results['bertscore']:.4f}")
    print("="*30)

    # Optionnel : Sauvegarder les prédictions dans un CSV pour analyse manuelle
    if OUTPUT_CSV:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df = pd.DataFrame({
            'Ground_Truth': results['ground_truths'],
            'Retrieval_Prediction': results['predictions']
        })
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nPrédictions sauvegardées dans : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()