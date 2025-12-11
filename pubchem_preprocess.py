
import torch
import os
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import pickle 
# --- 1. Définition de la classe du Dataset (comme dans votre code) ---
class PubChemDataset(InMemoryDataset):
    def __init__(self, path):
        super(PubChemDataset, self).__init__('./')
        self.data, self.slices = torch.load(path, weights_only=False)
        
    def __getitem__(self, idx):
        return self.get(idx)

# --- Fonctions utilitaires pour le chargement/enregistrement ---

def load_data_pt(file_path):
    """Charge les données depuis un fichier .pt PyTorch."""
    data = torch.load(file_path, weights_only=False)
    print(f"✅ Données chargées depuis : {file_path}")
    return data

def save_data_pt(data, file_path):
    """Sauvegarde les données au format .pt PyTorch avec compression."""
    # Utilise _use_new_zipfile_serialization pour une meilleure compression
    torch.save(data, file_path, _use_new_zipfile_serialization=True)
    
    # Affiche la taille du fichier
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
    4. Nettoie les attributs non essentiels (cid, etc.) pour éviter les conflits.
    """
    print("⏳ Démarrage du filtrage et du renommage du dataset PubChem...")
    
    filtered_data_list = []
    filter_string = "with data available"
    
    # Attributs essentiels à garder
    essential_attrs = ['x', 'edge_index', 'edge_attr', 'smiles', 'num_nodes']
    
    for i in tqdm(range(len(pubchem_dataset)), desc="Filtrage et nettoyage"):
        item = pubchem_dataset.get(i)
        
        if hasattr(item, 'text') and item.text is not None and filter_string not in item.text:
            
            # Renommer text -> description
            item.description = item.text
            del item.text
            
            # Nettoyer les attributs non essentiels (cid, etc.)
            attrs_to_remove = []
            for attr in item.keys():
                if attr not in essential_attrs and attr != 'description':
                    attrs_to_remove.append(attr)
            
            for attr in attrs_to_remove:
                delattr(item, attr)
            
            # S'assurer que les tenseurs sont sur CPU
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

# --- 3. Pipeline Principale ---

def main_pipeline():
    """
    Pipeline principale de fusion des datasets.
    Ajoute un attribut .id unique à chaque échantillon avant l'enregistrement.
    Nettoie tous les attributs non essentiels pour éviter les conflits de batching.
    """
    PUBCHEM_PT_PATH = './PubChem324kV2/pretrain.pt'
    TRAIN_DATA_PATH = 'src/data/train_graphs_smiles.pkl'
    OUTPUT_PT_PATH = 'src/data/full_train_fused.pt'
    
    if not os.path.exists(PUBCHEM_PT_PATH):
        print(f"❌ Erreur : Fichier PubChem non trouvé à {PUBCHEM_PT_PATH}. Veuillez vérifier le chemin.")
        return
    
    print(f"⏳ Chargement et traitement du dataset PubChem...")
    pubchem_dataset = PubChemDataset(PUBCHEM_PT_PATH)
    processed_pubchem_list = process_pubchem_dataset(pubchem_dataset)
    
    print(f"\n⏳ Chargement du dataset existant depuis {TRAIN_DATA_PATH}...")
    if TRAIN_DATA_PATH.endswith('.pkl'):
        with open(TRAIN_DATA_PATH, 'rb') as f:
            existing_train_list = pickle.load(f)
    else:
        existing_train_list = load_data_pt(TRAIN_DATA_PATH)
    
    print(f"Dataset existant : {len(existing_train_list)} échantillons.")

    # Nettoyer le dataset existant aussi pour uniformiser
    print(f"\n⏳ Nettoyage et optimisation du dataset existant...")
    essential_attrs = ['x', 'edge_index', 'edge_attr', 'smiles', 'num_nodes', 'description', 'id']
    
    for item in tqdm(existing_train_list, desc="Nettoyage"):
        if isinstance(item, Data):
            # Nettoyer les attributs non essentiels
            attrs_to_remove = []
            for attr in item.keys():
                if attr not in essential_attrs:
                    attrs_to_remove.append(attr)
            
            for attr in attrs_to_remove:
                delattr(item, attr)
            
            # Optimisation CPU
            if torch.is_tensor(item.x):
                item.x = item.x.cpu()
            if torch.is_tensor(item.edge_index):
                item.edge_index = item.edge_index.cpu()
            if hasattr(item, 'edge_attr') and torch.is_tensor(item.edge_attr):
                item.edge_attr = item.edge_attr.cpu()
            
            # S'assurer que smiles est une string
            if hasattr(item, 'smiles') and not isinstance(item.smiles, str):
                item.smiles = str(item.smiles)

    # Fusion
    fused_dataset = existing_train_list + processed_pubchem_list
    
    print(f"\n⏳ Ajout de l'attribut .id pour {len(fused_dataset)} échantillons...")
    
    for i in tqdm(range(len(fused_dataset)), desc="Indexation"):
        item = fused_dataset[i]
        
        # Ajouter un ID unique si pas déjà présent
        if not hasattr(item, 'id') or item.id is None:
            item.id = f'sample_{i}'
        
        # S'assurer que smiles est une string
        if hasattr(item, 'smiles') and not isinstance(item.smiles, str):
            item.smiles = str(item.smiles)

    print(f"\n✅ Fusion et nettoyage terminés. Taille totale : {len(fused_dataset)} échantillons.")

    print(f"\n⏳ Enregistrement du dataset fusionné en format .pt...")
    save_data_pt(fused_dataset, OUTPUT_PT_PATH)

if __name__ == '__main__':
    main_pipeline()