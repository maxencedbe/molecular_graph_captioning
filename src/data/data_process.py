import pickle
from typing import Dict, List, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

def load_data(file_path: str) -> List[Batch]:
    with open(file_path, 'rb') as f:
        graphs = pickle.load(f)
    return graphs

# =========================================================
# Feature maps for atom and bond attributes
# =========================================================

x_map: Dict[str, List[Any]] = {
    'atomic_num': list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER',
        'CHI_TETRAHEDRAL','CHI_ALLENE','CHI_SQUAREPLANAR','CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2','OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED','SINGLE','DOUBLE','TRIPLE','QUADRUPLE','QUINTUPLE','HEXTUPLE',
        'ONEANDAHALF','TWOANDAHALF','THREEANDAHALF','FOURANDAHALF','FIVEANDAHALF',
        'AROMATIC','IONIC','HYDROGEN','THREECENTER','DATIVEONE','DATIVE','DATIVEL',
        'DATIVER','OTHER','ZERO',
    ],
    'stereo': [
        'STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# =========================================================
# Load precomputed text embeddings
# =========================================================
def load_id2emb(csv_path: str) -> Dict[str, torch.Tensor]:
    """
    Load precomputed text embeddings from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: ID, embedding
                  where embedding is comma-separated floats
        
    Returns:
        Dictionary mapping ID (str) to embedding tensor
    """
    df = pd.read_csv(csv_path)
    id2emb = {}
    for _, row in df.iterrows():
        id_ = str(row["ID"])
        emb_str = row["embedding"]
        emb_vals = [float(x) for x in str(emb_str).split(',')]
        id2emb[id_] = torch.tensor(emb_vals, dtype=torch.float32)
    return id2emb


# =========================================================
# Load descriptions from preprocessed graphs
# =========================================================
def load_descriptions_from_graphs(graph_path: str) -> Dict[str, str]:
    """
    Load ID to description mapping from preprocessed graph file.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        
    Returns:
        Dictionary mapping ID (str) to description (str)
    """
    with open(graph_path, 'rb') as f:
        graphs = pickle.load(f)
    
    id2desc = {}
    for graph in graphs:
        id2desc[graph.id] = graph.description
    
    return id2desc

def embdict_to_tensor(emb_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert a dictionary of embeddings to a stacked tensor.
    Args:
        emb_dict: Dictionary mapping ID (str) to embedding tensor
        id_list: List of IDs (str) to retrieve embeddings for
        
    Returns:
        Stacked tensor of embeddings corresponding to the IDs in id_list
    """
    emb_list = [emb_dict[id_] for id_ in emb_dict.keys()]
    return torch.stack(emb_list, dim=0)
# =========================================================
# One hot encode the features
# =========================================================
def ohe_node_features(graph):
    n_node = graph.x.size(0)
    total_feat = sum([len(l) for l in x_map.values()])
    final_x = torch.zeros((n_node, total_feat))
    
    offset = 0
    for i, feat in enumerate(x_map.keys()):
        l_feat = x_map[feat]
        tensor_feat = torch.zeros((n_node, len(l_feat)))
        
        indices = graph.x[:, i].long()  
        tensor_feat[torch.arange(n_node), indices] = 1.0
        
        final_x[:, offset:offset + len(l_feat)] = tensor_feat
        offset += len(l_feat)
    final_x = final_x.float()
    graph.x = final_x
    return graph

def ohe_edge_features(graph):
    n_edge = graph.edge_attr.size(0)
    total_feat = sum([len(l) for l in e_map.values()])
    final_attr = torch.zeros((n_edge, total_feat))
    
    offset = 0
    for i, feat in enumerate(e_map.keys()):
        l_feat = e_map[feat]
        tensor_feat = torch.zeros((n_edge, len(l_feat)))
        
        indices = graph.edge_attr[:, i].long()  
        tensor_feat[torch.arange(n_edge), indices] = 1.0
        
        final_attr[:, offset:offset + len(l_feat)] = tensor_feat
        offset += len(l_feat)
    final_attr = final_attr.float()
    graph.edge_attr = final_attr
    return graph

from transformers import BertTokenizer
TEXT_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL_NAME)

def tokenize_descriptions(descriptions):
    encoded = tokenizer(descriptions, padding=True, truncation=True, return_tensors='pt')
    return encoded

# =========================================================
# Dataset that loads preprocessed graphs and text embeddings
# =========================================================
class PreprocessedGraphDataset(Dataset):
    """
    Dataset that loads pre-saved molecule graphs with optional text embeddings.
    
    Args:
        graph_path: Path to .pkl file containing list of pre-saved graphs
        emb_dict: Dictionary mapping ID to text embedding tensors (optional)
        encode_feat: whether to encode the features or not (OHE)
    """
    def __init__(self, graph_path: str, 
                 emb_dict: Dict[str, torch.Tensor] = None,
                 encode_feat: bool = True):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
        self.emb_dict = emb_dict
        self.ids = [g.id for g in self.graphs]
        self.encode_feat = encode_feat
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.encode_feat : 
            graph = ohe_node_features(graph)
            graph = ohe_edge_features(graph)
        if self.emb_dict is not None:
            id_ = graph.id
            text_emb = self.emb_dict[id_]
            return graph, text_emb
        else:
            return graph, graph.description


def collate_fn(batch):
    """
    Collate function for DataLoader to batch graphs with optional text embeddings.
    
    Args:
        batch: List of graph Data objects or (graph, text_embedding) tuples
        
    Returns:
        Batched graph or (batched_graph, stacked_text_embeddings)
    """
    if isinstance(batch[0], tuple):
        graphs, descriptions = zip(*batch)
        batch_graph = Batch.from_data_list(list(graphs))
        batch_description = tokenize_descriptions(descriptions)
        return batch_graph, batch_description
    else:
        return Batch.from_data_list(batch)