
from src.utils import retrieve_captioning_topk
from src.data.data_process import embdict_to_tensor, load_id2emb
import torch
import pandas as pd
import pickle
import json
import tqdm
top_k = 25

file_path = "src/data/train_embeddings_bge.csv"
df = pd.read_csv(file_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_emb = load_id2emb(file_path)
train_caption_tensor = embdict_to_tensor(train_emb).to(device)

print(f"Lecture du fichier : {file_path}")

with open("src/data/train_graphs_smiles.pkl", 'rb') as f:
    train_graphs = pickle.load(f)

json_list = []

for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    mol_id = str(row["ID"])
    emb_str = row["embedding"]
    emb_vals = [float(x) for x in str(emb_str).split(',')]
    emb = torch.tensor(emb_vals, dtype=torch.float32).unsqueeze(0).to(device)
    neg = retrieve_captioning_topk(emb,train_caption_tensor,top_k=top_k)
    neg = neg.squeeze(0).cpu().numpy().tolist()
    neg.pop(0)
    hard_negatives = [train_graphs[i].description for i in neg]
    smile = train_graphs[int(mol_id)].smiles
    pos_caption = train_graphs[int(mol_id)].description
    json_row = {
        "query": smile,
        "pos": [pos_caption],
        "neg": hard_negatives,
        "prompt": "Represent this sentence for searching relevant passages:"
    }
    json_list.append(json_row)

with open("src/data/train_bge_sft.jsonl", 'w') as f:
    for item in json_list:
        f.write(json.dumps(item) + '\n')
