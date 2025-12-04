from data.data_process import load_data, ohe_edge_features, ohe_node_features
from model.model import GEncoder
import torch


if __name__ == "__main__":
    data_file = "data/test_graphs.pkl"
    data = load_data(data_file)
    
    print("Data loaded successfully:")
    print(data[0])
    
    model = GEncoder()
    print("Model instantiated successfully:")
    print(model)
    sample_graph = data[0]
    sample_graph = ohe_node_features(sample_graph)
    sample_graph = ohe_edge_features(sample_graph)
    sample_graph.batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)  
    output = model(sample_graph)
    print("Model forward pass successful. Output:")
    print(output.shape)
