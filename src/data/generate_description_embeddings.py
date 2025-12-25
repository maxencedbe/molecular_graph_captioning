#!/usr/bin/env python3
import pickle
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os 
import numpy as np

# Configuration
MAX_TOKEN_LENGTH = 512 
MODEL_NAME = "tencent/KaLM-Embedding-Gemma3-12B-2511"
LOCAL_CACHE = "./model_cache" # Dossier local pour le modèle
MRL_DIM = 1024 # Dimension optimale pour le retrieval avec ce modèle

os.makedirs(LOCAL_CACHE, exist_ok=True)

print(f"Loading KaLM model: {MODEL_NAME}...")

model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    cache_folder=LOCAL_CACHE, # Définit le cache en local
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
    }
)
model.max_seq_length = MAX_TOKEN_LENGTH

try:
    with open('train_graphs_selfies.pkl', 'rb') as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} graphs")
except FileNotFoundError:
    print("❌ Erreur: 'train_graphs_selfies.pkl' non trouvé.")
    exit()

ids = []
descriptions = [] 
for graph in graphs:
    ids.append(getattr(graph, 'id', f'unknownid{len(ids)}')) 
    descriptions.append(getattr(graph, 'description', '')) 

print(f"\nGenerating embeddings (Dim: {MRL_DIM})...")

embeddings_array = model.encode(
    descriptions, 
    batch_size=8, # Réduit pour éviter OOM sur 12B
    show_progress_bar=True,
    convert_to_numpy=True
)

# Application du Slicing MRL et Re-normalisation
# Essentiel pour maintenir la précision du retrieval en dimension réduite
embeddings_array = embeddings_array[:, :MRL_DIM]
norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
embeddings_array = np.divide(embeddings_array, norms, out=np.zeros_like(embeddings_array), where=norms!=0)

result = pd.DataFrame({
    'ID': ids,
    'embedding': [','.join(map(str, emb)) for emb in embeddings_array] 
})

output_path = './train_embeddings_kalm_1024.csv'
result.to_csv(output_path, index=False)

print(f"\nSaved {len(embeddings_array)} embeddings to {output_path}")
print(f"La taille finale de l'embedding est : {embeddings_array.shape[1]}")
print("\nDone!")