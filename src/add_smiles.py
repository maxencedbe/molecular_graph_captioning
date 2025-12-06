import torch_geometric.utils.smiles as tosmiles
from src.data.data_process import load_data
import pickle 
import tqdm

dataset_val = load_data("src/data/validation_graphs.pkl")
dataset_train = load_data("src/data/train_graphs.pkl")
dataset_test = load_data("src/data/test_graphs.pkl")


for dataset in [dataset_val, dataset_train, dataset_test]:

    for graph in tqdm.tqdm(dataset, desc="Converting to SMILES"):
        smiles = tosmiles.to_smiles(graph)
        graph.smiles = smiles


with open("src/data/validation_graphs_smiles.pkl", "wb") as f:
    pickle.dump(dataset_val, f)

with open("src/data/train_graphs_smiles.pkl", "wb") as f:
    pickle.dump(dataset_train, f)

with open("src/data/test_graphs_smiles.pkl", "wb") as f:
    pickle.dump(dataset_test, f)
