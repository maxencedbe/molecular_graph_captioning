#!/usr/bin/env python3
"""Generate BGE embeddings for molecular descriptions."""

import pickle
import pandas as pd
import torch
from FlagEmbedding import FlagModel # Import du modèle BGE natif
from tqdm import tqdm
import os 
import numpy as np # Ajout de numpy pour la conversion en array

MAX_TOKEN_LENGTH = 512 

#MODEL_NAME = 'BAAI/bge-large-en-v1.5'
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
print(f"Loading BGE model: {MODEL_NAME}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FlagModel(
    MODEL_NAME, 
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True, # Décommenter si la mémoire GPU est limitée
    device=str(device)
)

print(f"Model loaded on: {device}")

try:
    print(f"Loading graphs from train_graphs.pkl...")
    with open('train_graphs.pkl', 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")
except FileNotFoundError:
    print("❌ Erreur: 'train_graphs.pkl' non trouvé. Veuillez vérifier le chemin.")
    exit()
ids = []
descriptions = [] 

for graph in tqdm(graphs, desc="Extraction des descriptions", total=len(graphs)):
    # Assurer la robustesse en cas d'attribut manquant
    ids.append(getattr(graph, 'id', f'unknownid{len(ids)}')) 
    descriptions.append(getattr(graph, 'description', '')) 

print("\nGenerating embeddings (BGE batch encoding)...")

embeddings_array = model.encode(
    descriptions, 
    batch_size=32, # Ajustez pour votre GPU
    max_length=MAX_TOKEN_LENGTH, # Ajout de la limite max
)


result = pd.DataFrame({
    'ID': ids,
    # Assurez-vous que l'objet est bien un array NumPy avant la conversion en chaîne
    'embedding': [','.join(map(str, emb)) for emb in embeddings_array] 
})
output_path = f'./train_embeddings_bge.csv' # Nom mis à jour pour BGE
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result.to_csv(output_path, index=False)
print(f"\nSaved {len(embeddings_array)} embeddings to {output_path}")
print(f"La taille de l'embedding est : {embeddings_array.shape[1]}")

print("\nDone!")