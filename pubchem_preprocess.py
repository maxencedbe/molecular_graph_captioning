import torch
import pickle
import os
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__('./')
        self.data, self.slices = torch.load(path, weights_only=False)
        
    def __getitem__(self, idx):
        return self.get(idx)


def load_data_pt(file_path):
    """Charge les données depuis un fichier .pt PyTorch."""
    data = torch.load(file_path, weights_only=False)
    print(f"✅ Données chargées depuis : {file_path}")
    return data

def save_data_pt(data, file_path):
    """Sauvegarde les données au format .pt PyTorch avec compression."""
    torch.save(data, file_path, _use_new_zipfile_serialization=True)
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"✅ Données sauvegardées dans : {file_path}")
    print(f"📊 Taille du fichier : {size_mb:.2f} Mo")

# --- 2. Fonctions de Traitement ---

def process_pubchem_dataset(pubchem_dataset):
    """
    Traite le dataset PubChem :
    1. Filtre les échantillons contenant "with data available" dans .text.
    2. Remplace l'attribut .text par .description.
    3. Garde le format Data pour compatibilité.
    """
    print("⏳ Démarrage du filtrage et du renommage du dataset PubChem...")
    
    filtered_data_list = []
    filter_string = "with data available"
    
    for i in tqdm(range(len(pubchem_dataset)), desc="Filtrage et renommage"):
        item = pubchem_dataset.get(i)
        
        if hasattr(item, 'text') and item.text is not None and filter_string not in item.text:
            
            item.description = item.text
            del item.text
            
            if torch.is_tensor(item.x):
                item.x = item.x.cpu()
            if torch.is_tensor(item.edge_index):
                item.edge_index = item.edge_index.cpu()
            if hasattr(item, 'edge_attr') and torch.is_tensor(item.edge_attr):
                item.edge_attr = item.edge_attr.cpu()
            
            if hasattr(item, 'smiles') and item.smiles is not None:
                filtered_data_list.append(item)
    
    print(f"Dataset PubChem initial : {len(pubchem_dataset)} échantillons.")
    print(f"Dataset PubChem filtré : {len(filtered_data_list)} échantillons.")
    return filtered_data_list


def main_pipeline():
    """
    Pipeline principale de fusion des datasets.
    Ajoute un attribut .id unique à chaque échantillon avant l'enregistrement.
    """
    PUBCHEM_PT_PATH = './PubChem324kV2/pretrain.pt'
    TRAIN_DATA_PATH = 'src/data/train_graphs_smiles.pkl'
    OUTPUT_PT_PATH = 'src/data/full_train_fused_indexed.pt'
    
    if not os.path.exists(PUBCHEM_PT_PATH):
        print(f"❌ Erreur : Fichier PubChem non trouvé à {PUBCHEM_PT_PATH}. Veuillez vérifier le chemin.")
        return
    
    print(f"⏳ Chargement et traitement du dataset PubChem...")
    pubchem_dataset = PubChemDataset(PUBCHEM_PT_PATH)
    processed_pubchem_list = process_pubchem_dataset(pubchem_dataset)
    
    print(f"\n⏳ Chargement du dataset existant depuis {TRAIN_DATA_PATH}...")
    if TRAIN_DATA_PATH.endswith('.pkl'):
        import pickle
        with open(TRAIN_DATA_PATH, 'rb') as f:
            existing_train_list = pickle.load(f)
    else:
        existing_train_list = load_data_pt(TRAIN_DATA_PATH)
    
    print(f"Dataset existant : {len(existing_train_list)} échantillons.")

    print(f"\n⏳ Optimisation CPU du dataset existant...")
    for item in tqdm(existing_train_list, desc="Optimisation CPU"):
        if isinstance(item, Data):
            if torch.is_tensor(item.x):
                item.x = item.x.cpu()
            if torch.is_tensor(item.edge_index):
                item.edge_index = item.edge_index.cpu()
            if hasattr(item, 'edge_attr') and torch.is_tensor(item.edge_attr):
                item.edge_attr = item.edge_attr.cpu()

    fused_dataset = existing_train_list + processed_pubchem_list
    
    print(f"\n⏳ Ajout de l'attribut .id (Réindexation) pour {len(fused_dataset)} échantillons...")
    
    for i in tqdm(range(len(fused_dataset)), desc="Indexation"):
        item = fused_dataset[i]

        item_id = f'i' 
        
        item.id = item_id 
        
        if hasattr(item, 'smiles') and not isinstance(item.smiles, str):
            item.smiles = str(item.smiles)

    print(f"\n✅ Fusion et Indexation terminées. Taille totale : {len(fused_dataset)} échantillons.")

    print(f"\n⏳ Enregistrement du dataset fusionné en format .pt...")
    save_data_pt(fused_dataset, OUTPUT_PT_PATH)

if __name__ == '__main__':
    main_pipeline()