import os
import pickle
import tqdm
import torch_geometric.utils.smiles as tosmiles
from src.data.data_process import load_data

data_files = [
    ("src/data/train_graphs.pkl", "src/data/train_graphs_smiles.pkl"),
    ("src/data/validation_graphs.pkl", "src/data/validation_graphs_smiles.pkl"),
    ("src/data/test_graphs.pkl", "src/data/test_graphs_smiles.pkl")
]

for input_path, output_path in data_files:
    dataset = load_data(input_path)
    output_dataset = []

    for graph in tqdm.tqdm(dataset, desc=f"Processing {os.path.basename(input_path)}"):
        try:
            # Conversion du graphe PyG en SMILES
            smiles = tosmiles.to_smiles(graph)
            
            if smiles:
                graph.smiles = smiles
                output_dataset.append(graph)
        
        except Exception:
            # On ignore les graphes qui posent problème lors de la conversion
            continue

    # Sauvegarde du nouveau dataset contenant les SMILES
    with open(output_path, "wb") as f:
        pickle.dump(output_dataset, f)

print("Traitement terminé. Les fichiers SMILES sont prêts.")