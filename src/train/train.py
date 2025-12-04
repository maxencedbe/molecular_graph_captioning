import torch
import tqdm
from src.utils import contrastive_loss

epochs = 50
batch_size = 32
learning_rate = 5e-6
weight_decay = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm.tqdm(dataloader, desc="Training Epoch", leave=False)
    for batch_graph, batch_text_emb in progress_bar:
        batch_graph = batch_graph.to(device)
        batch_text_emb = batch_text_emb.to(device)

        optimizer.zero_grad()

        z_graph = model(batch_graph)

        loss = contrastive_loss(z_graph, batch_text_emb)

        loss.backward()

        optimizer.step()
        current_loss_value = loss.item()
        
        total_loss += loss.item() * batch_graph.num_graphs
        progress_bar.set_postfix(loss=f'{current_loss_value:.4f}')

    return total_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_graph, batch_text_emb in dataloader:
            batch_graph = batch_graph.to(device)
            batch_text_emb = batch_text_emb.to(device)

            z_graph = model(batch_graph)

            loss = contrastive_loss(z_graph, batch_text_emb)

            total_loss += loss.item() * batch_graph.num_graphs

    return total_loss / len(dataloader.dataset)


def main():
    from src.data.data_process import load_id2emb, PreprocessedGraphDataset, collate_fn
    from torch.utils.data import DataLoader
    from src.model.model import GEncoder
    import torch.optim as optim

    train_data_file = "data/train_graphs.pkl"
    val_data_file = "data/validation_graphs.pkl"
    train_emb_csv = "src/data/train_embeddings.csv"
    val_emb_csv   = "src/data/validation_embeddings.csv"

    train_emb = load_id2emb(train_emb_csv)
    val_emb = load_id2emb(val_emb_csv)

    train_dataset = PreprocessedGraphDataset(train_data_file,train_emb, encode_feat=True)
    val_dataset = PreprocessedGraphDataset(val_data_file,val_emb, encode_feat=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()