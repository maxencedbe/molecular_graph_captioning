import torch
import pandas as pd

from src.utils import retrieve_captioning
from src.data.data_process import load_data, embdict_to_tensor, load_id2emb, PreprocessedGraphDataset, collate_fn
from src.model.model_gin import GEncoder
from torch.utils.data import DataLoader

model_path = "src/saved_model/best_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GEncoder().to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def main():
    test_data_file = "src/data/test_graphs.pkl"

    test_dataset = PreprocessedGraphDataset(test_data_file, encode_feat=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_pred_captions = []

    train_data_file = "src/data/train_graphs.pkl"
    train_emb_csv = "src/data/train_embeddings_bge.csv"

    train_emb = load_id2emb(train_emb_csv)
    train_data = load_data(train_data_file)
    test_data = load_data(test_data_file)
    train_caption_tensor = embdict_to_tensor(train_emb)

    ids = [graph.id for graph in test_data]

    with torch.no_grad():
        for batch_graph in test_loader:
            batch_graph = batch_graph.to(device)
            z_graph, _, _ = model(batch_graph)

            text_id = retrieve_captioning(z_graph, train_caption_tensor.to(device))
            pred_caption = [train_data[i].description for i in text_id.cpu().numpy()]

            all_pred_captions.extend(pred_caption)

    df = pd.DataFrame({'ID': ids, 'description': all_pred_captions})
    df.to_csv("src/inference/predicted_captions.csv", index=False)

if __name__ == "__main__":
    main()