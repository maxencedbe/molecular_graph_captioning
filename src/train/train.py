import torch
import tqdm
import wandb
import os
import torch.optim as optim
import torch.nn.functional as F

from src.utils import contrastive_loss, MolecularCaptionEvaluator, retrieve_captioning 
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn, load_id2emb, embdict_to_tensor
from torch.utils.data import DataLoader
from src.model.model import GEncoder, node_feat_dim, hidden_dim
from src.model.x_model import MoLCABackbone

epochs = 50
batch_size = 32
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
    
    for batch_idx, (batch_graph, batch_text) in enumerate(progress_bar):
        batch_graph = batch_graph.to(device)
        batch_text = batch_text.to(device)
        batch_size = batch_graph.num_graphs
        
        optimizer.zero_grad()

        # Scores positifs (paires correctes)
        positive_scores = model(batch_graph, batch_text) 
        
        perm = torch.randperm(batch_size, device=device)
        while (perm == torch.arange(batch_size, device=device)).any():
            perm = torch.randperm(batch_size, device=device)
        
        batch_text_neg = {k: v[perm] for k, v in batch_text.items()}
        negative_scores = model(batch_graph, batch_text_neg) 
        
        # Hinge loss: max(0, margin - (positive_score - negative_score))
        margin = 0.5
        loss = torch.clamp(margin - (positive_scores - negative_scores), min=0).mean()

        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item() * batch_size
        total_loss += batch_loss
        num_samples_processed += batch_size
        
        running_loss = total_loss / num_samples_processed
        progress_bar.set_postfix(loss=f'{running_loss:.4f}')
        
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/step": epoch * len(dataloader) + batch_idx
        })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

from transformers import BertTokenizer
def validate_epoch(model, dataloader, val_data_list, evaluator, device):
    model.eval()
    total_loss = 0
    total_score = 0
    num_samples_processed = 0
    
    # Pré-calculer tous les embeddings de texte de validation
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_text_inputs = [data.description for data in val_data_list]
    text_encoded = tokenizer(all_text_inputs, padding=True, truncation=True, return_tensors='pt')
    text_encoded = {k: v.to(device) for k, v in text_encoded.items()}
    
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_graph, batch_text in progress_bar:
            batch_graph = batch_graph.to(device)
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
            batch_size = batch_graph.num_graphs
            
            # Scores positifs
            positive_scores = model(batch_graph, batch_text)
            
            # Créer négatifs pour la loss
            perm = torch.randperm(batch_size, device=device)
            while (perm == torch.arange(batch_size, device=device)).any():
                perm = torch.randperm(batch_size, device=device)
            
            batch_text_neg = {k: v[perm] for k, v in batch_text.items()}
            negative_scores = model(batch_graph, batch_text_neg)
            
            margin = 1.0
            loss = torch.clamp(margin - (positive_scores - negative_scores), min=0).mean()
            total_loss += loss.item() * batch_size
            
            # Retrieval : comparer chaque molécule avec TOUS les textes de validation
            pred_caption = []
            for i in range(batch_size):
                single_graph = batch_graph[i]
                
                # Calculer le score avec tous les textes de validation
                scores = []
                for j in range(len(val_data_list)):
                    text_j = {k: v[j:j+1] for k, v in text_encoded.items()}
                    score = model(single_graph, text_j)
                    scores.append(score.item())
                
                # Trouver l'index du texte avec le meilleur score
                predicted_idx = torch.tensor(scores).argmax().item()
                pred_caption.append(val_data_list[predicted_idx].description)
            
            # True captions du batch
            individual_graphs = batch_graph.to_data_list()
            true_caption = [graph.description for graph in individual_graphs]
            
            # Évaluer avec l'evaluator
            score = evaluator.evaluate_batch(pred_caption, true_caption)
            total_score += score['composite_score'] * batch_size
            
            num_samples_processed += batch_size
            
            progress_bar.set_postfix(
                v_loss=f'{total_loss / num_samples_processed:.4f}',
                v_score=f'{total_score / num_samples_processed:.4f}'
            )

    avg_loss = total_loss / num_samples_processed
    avg_score = total_score / num_samples_processed
    return avg_loss, avg_score


def main():
    train_data_file = "src/data/train_graphs.pkl"
    val_data_file = "src/data/validation_graphs.pkl"
    train_emb_csv = "src/data/train_embeddings.csv"
    val_emb_csv   = "src/data/validation_embeddings.csv"

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

    print("Loading data...")

    train_emb = load_id2emb(train_emb_csv)
    val_emb = load_id2emb(val_emb_csv)
    train_data = load_data(train_data_file)
    val_data_list = load_data(val_data_file)

    val_caption_tensor = embdict_to_tensor(train_emb).to(device)

    train_dataset = PreprocessedGraphDataset(train_data_file, encode_feat=True)
    val_dataset = PreprocessedGraphDataset(val_data_file, val_emb, encode_feat=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MoLCABackbone().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    evaluator = MolecularCaptionEvaluator(device=device)

    wandb.watch(model, log="all", log_freq=100)

    best_score = 0.0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        wandb.log({
            "train/epoch_loss": train_loss,
            "epoch": epoch
        })
        
        if (epoch) % val_freq == 0:
            val_loss, score = validate_epoch(
                model, 
                val_loader, 
                val_caption_tensor, 
                train_data,
                evaluator,
                device
            )
            
            wandb.log({
                "val/loss": val_loss,
                "val/composite_score": score,
                "epoch": epoch
            })
            
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

    wandb.finish()


if __name__ == "__main__":
    main()