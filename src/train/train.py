import torch

from utils import contrastive_loss

epochs = 50
batch_size = 32
learning_rate = 1e-3
weight_decay = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_graph, batch_text_emb in dataloader:
        batch_graph = batch_graph.to(device)
        batch_text_emb = batch_text_emb.to(device)

        optimizer.zero_grad()

        z_graph = model(batch_graph)

        loss = contrastive_loss(z_graph, batch_text_emb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * batch_graph.num_graphs

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