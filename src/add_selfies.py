import os
import pickle
import tqdm
import selfies as sf
import torch_geometric.utils.smiles as tosmiles
from src.data.data_process import load_data

# Définition des fichiers d'entrée et de sortie (suffixe _selfies)
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
            # 1. Conversion intermédiaire : Graphe -> SMILES
            # (Nécessaire car selfies a besoin d'un SMILES en entrée)
            smiles = tosmiles.to_smiles(graph)
            
            if not smiles:
                continue

            # 2. Conversion finale : SMILES -> SELFIES
            selfies_str = sf.encoder(smiles)

            if selfies_str:
                # On ne stocke que le SELFIES comme demandé
                graph.selfies = selfies_str
                output_dataset.append(graph)
        
        except Exception:
            # On ignore les erreurs de conversion (molécules invalides)
            continue

    # Sauvegarde du dataset avec les attributs SELFIES
    with open(output_path, "wb") as f:
        pickle.dump(output_dataset, f)

print("Traitement terminé. Les fichiers SELFIES sont prêts.")