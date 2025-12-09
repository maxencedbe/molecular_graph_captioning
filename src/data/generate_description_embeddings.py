#!/usr/bin/env python3
"""Generate BERT embeddings for molecular descriptions."""

import pickle
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration
MAX_TOKEN_LENGTH = 128

# Load BERT model
print("Loading BERT model...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
print(f"Model loaded on: {device}")

# Process each split


# Load graphs from pkl file
pt_path = f'full_train_fused.pt' 
print(f"Loading from {pt_path}...")

# Utilisation de torch.load() à la place de pickle.load()
# L'argument 'weights_only=False' assure que les objets Data complexes sont chargés correctement.
graphs = torch.load(pt_path, weights_only=False)

print(f"Loaded {len(graphs)} graphs")

# Generate embeddings
ids = []
embeddings = []

for graph in tqdm(graphs, total=len(graphs)):
    # Get description from graph
    description = graph.description
    
    # Tokenize
    inputs = tokenizer(description, return_tensors='pt', 
                        truncation=True, max_length=MAX_TOKEN_LENGTH, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embedding
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    ids.append(graph.id)
    embeddings.append(embedding)

# Save to CSV
result = pd.DataFrame({
    'ID': ids,
    'embedding': [','.join(map(str, emb)) for emb in embeddings]
})
output_path = f'data/train_fused_embeddings.csv'
result.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

print("\nDone!")

