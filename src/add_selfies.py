import os
import pickle
import tqdm
import selfies as sf
import torch_geometric.utils.smiles as tosmiles
from src.data.data_process import load_data

data_files = [
    ("src/data/train_graphs.pkl", "src/data/train_graphs_selfies.pkl"),
    ("src/data/validation_graphs.pkl", "src/data/validation_graphs_selfies.pkl"),
    ("src/data/test_graphs.pkl", "src/data/test_graphs_selfies.pkl")
]

for input_path, output_path in data_files:
    dataset = load_data(input_path)
    output_dataset = []

    for graph in tqdm.tqdm(dataset, desc=f"Processing {os.path.basename(input_path)}"):
        try:
            smiles = tosmiles.to_smiles(graph)
            if not smiles:
                continue

            selfies_str = sf.encoder(smiles)
            if not selfies_str:
                continue
        
            graph.smiles = smiles
            graph.selfies = selfies_str
            output_dataset.append(graph)
        
        except Exception:
            continue

    with open(output_path, "wb") as f:
        pickle.dump(output_dataset, f)
