import torch
import tqdm
import wandb
from src.utils import retrieve_captioning, contrastive_loss, MolecularCaptionEvaluator
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn, embdict_to_tensor
from torch.utils.data import DataLoader
from src.model.model import GEncoder,GEncoder2
import torch.optim as optim
import torch.nn.functional as F

epochs = 500
batch_size = 64
learning_rate = 5e-4
weight_decay = 1e-5
val_freq = 5
save_freq = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_samples_processed = 0
    progress_bar = tqdm.tqdm(dataloader, desc="Training Epoch", leave=False)
    
    for batch_idx, (batch_graph, batch_text_emb) in enumerate(progress_bar):
        batch_graph = batch_graph.to(device)
        batch_text_emb = batch_text_emb.to(device)

        optimizer.zero_grad()

        z_graph = model(batch_graph)

        loss = contrastive_loss(z_graph, batch_text_emb)

        loss.backward()

        optimizer.step()
        
        batch_loss = loss.item() * batch_graph.num_graphs
        total_loss += batch_loss
        num_samples_processed += batch_graph.num_graphs
        
        running_loss = total_loss / num_samples_processed
        progress_bar.set_postfix(loss=f'{running_loss:.4f}')
        
        # Log batch metrics to W&B
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/step": epoch * len(dataloader) + batch_idx
        })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate_epoch(model, dataloader, caption_emb, train_data, device):
    model.eval()
    total_loss = 0
    total_score = 0
    
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_graph, batch_text_emb in progress_bar:
            batch_graph = batch_graph.to(device)
            batch_text_emb = batch_text_emb.to(device)

            z_graph = model(batch_graph)

            loss = contrastive_loss(z_graph, batch_text_emb)
            
            caption_emb = caption_emb.to(device)
            text_id = retrieve_captioning(z_graph, caption_emb)
            pred_caption = [train_data[i].description for i in text_id.cpu().numpy()]

            individual_graphs = batch_graph.to_data_list()
            true_caption = [graph.description for graph in individual_graphs]
            
            score = MolecularCaptionEvaluator().evaluate_batch(pred_caption, true_caption)
            
            total_score += score['composite_score'] * batch_graph.num_graphs
            total_loss += loss.item() * batch_graph.num_graphs
            
            # Calculer les moyennes courantes
            avg_loss = total_loss / (progress_bar.n * dataloader.batch_size + batch_graph.num_graphs)
            avg_score = total_score / (progress_bar.n * dataloader.batch_size + batch_graph.num_graphs)
            
            progress_bar.set_postfix(loss=f'{avg_loss:.4f}', score=f'{avg_score:.4f}')

    avg_loss = total_loss / len(dataloader.dataset)
    avg_score = total_score / len(dataloader.dataset)
    return avg_loss, avg_score


def main():
    from src.data.data_process import load_id2emb, PreprocessedGraphDataset, collate_fn
    from torch.utils.data import DataLoader
    from src.model.model import GEncoder
    from src.model.ref_model import MolGNN
    import torch.optim as optim

    # Initialize W&B
    wandb.init(
        project="molecular-captioning",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "architecture": "GEncoder",
            "device": str(device)
        }
    )

    train_data_file = "src/data/train_graphs.pkl"
    val_data_file = "src/data/validation_graphs.pkl"
    train_emb_csv = "src/data/train_embeddings.csv"
    val_emb_csv   = "src/data/validation_embeddings.csv"

    train_emb = load_id2emb(train_emb_csv)
    val_emb = load_id2emb(val_emb_csv)
    train_data = load_data(train_data_file)
    train_caption_tensor = embdict_to_tensor(train_emb)

    train_dataset = PreprocessedGraphDataset(train_data_file, train_emb, encode_feat=True)
    val_dataset = PreprocessedGraphDataset(val_data_file, val_emb, encode_feat=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GEncoder().to(device)
    #model = MolGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Watch model with W&B
    wandb.watch(model, log="all", log_freq=100)

    best_score = 0.0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Log training metrics
        wandb.log({
            "train/epoch_loss": train_loss,
            "epoch": epoch
        })
        
        if (epoch) % val_freq == 0:
            val_loss, score = validate_epoch(model, val_loader, train_caption_tensor, train_data, device)
            
            # Log validation metrics
            wandb.log({
                "val/loss": val_loss,
                "val/composite_score": score,
                "epoch": epoch
            })
            
            # Save best model
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "src/saved_model/best_model.pth")
                wandb.save("src/saved_model/best_model.pth")
                wandb.run.summary["best_score"] = best_score
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Score: {score:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
            
        if (epoch+1) % save_freq == 0:
            checkpoint_path = f"src/saved_model/model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()