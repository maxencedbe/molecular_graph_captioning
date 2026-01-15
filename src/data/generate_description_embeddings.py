import pickle
import pandas as pd
import torch
from FlagEmbedding import FlagModel
from tqdm import tqdm
import os 
import numpy as np

MAX_TOKEN_LENGTH = 512 
MODEL_NAME = 'BAAI/bge-large-en-v1.5'
DATA_DIR = 'src/data'
FILES_TO_PROCESS = ['train_graphs.pkl', 'validation_graphs.pkl', 'test_graphs.pkl']

print(f"Loading BGE model: {MODEL_NAME}...")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

model = FlagModel(
    MODEL_NAME, 
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True,
    device=str(device)
)
print(f"Model loaded on: {device}")

for filename in FILES_TO_PROCESS:
    input_path = os.path.join(DATA_DIR, filename)
    
    output_name = filename.replace('_graphs.pkl', '_embeddings.csv')
    output_path = os.path.join(DATA_DIR, output_name)

    print(f"\n--- Processing: {filename} ---")

    try:
        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs from {input_path}")
    except FileNotFoundError:
        print(f"Warning: '{input_path}' not found. Skipping to the next file.")
        continue

    ids = []
    descriptions = [] 

    for graph in tqdm(graphs, desc=f"Extraction ({filename})", total=len(graphs)):
        ids.append(getattr(graph, 'id', f'unknownid{len(ids)}')) 
        descriptions.append(str(getattr(graph, 'description', ''))) 

    print(f"Generating embeddings for {filename}...")
    
    embeddings_array = model.encode(
        descriptions, 
        batch_size=32,
        max_length=MAX_TOKEN_LENGTH,
    )

    result = pd.DataFrame({
        'ID': ids,
        'embedding': [','.join(map(str, emb)) for emb in embeddings_array] 
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    
    print(f"Success: {len(embeddings_array)} embeddings saved to {output_path}")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")

print("\nAll tasks completed!")