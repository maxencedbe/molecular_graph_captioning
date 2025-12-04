from data.data_process import load_data,ohe_node_features,ohe_edge_features


if __name__ == "__main__":
    data_file = "src/data/test_graphs.pkl"
    data = load_data(data_file)
    print("Data loaded successfully:")
    print(data[0])
    print(ohe_node_features(data[0]))
    print(ohe_edge_features(data[0]))
    print(type(data[0]))