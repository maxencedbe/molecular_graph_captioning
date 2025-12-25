import torch
import tqdm
import wandb
import os
import torch.optim as optim
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup ,get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup 

from src.utils import contrastive_loss, siglip_loss_sampling, contrastive_loss_softmax_plus_hardneg, contrastive_loss_sampling, MolecularCaptionEvaluator, retrieve_captioning 
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn, load_id2emb, embdict_to_tensor
from torch.utils.data import DataLoader
from src.model.model_gat import GEncoder

epochs = 200
batch_size = 256
learning_rate = 5e-4
weight_decay = 1e-5
val_freq = 5
save_freq = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_epoch(model, dataloader, train_caption_tensor, optimizer, scheduler, device, epoch): 
    model.train()
    total_loss = 0
    num_samples_processed = 0
    progress_bar = tqdm.tqdm(dataloader, desc="Training Epoch", leave=False)
    
    for batch_idx, (batch_graph, batch_text_emb) in enumerate(progress_bar):
        batch_graph = batch_graph.to(device)
        batch_text_emb = batch_text_emb.to(device)

        optimizer.zero_grad()

        z_graph, _, _ = model(batch_graph)
        loss = contrastive_loss_sampling(z_graph, batch_text_emb, batch_idx, train_caption_tensor, batch_size=256)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        batch_loss = loss.item() * batch_graph.num_graphs
        total_loss += batch_loss
        num_samples_processed += batch_graph.num_graphs
        
        running_loss = total_loss / num_samples_processed
        progress_bar.set_postfix(loss=f'{running_loss:.4f}')
        
        wandb.log({
            "train/batch_loss": loss.item(),
            "train/learning_rate": scheduler.get_last_lr()[0], 
            "train/step": epoch * len(dataloader) + batch_idx
        })

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate_epoch(model, dataloader, val_caption_tensor, val_data_list, evaluator, device):
    model.eval()
    total_loss = 0
    total_score = 0
    total_bleu = 0
    total_bert = 0
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_graph, batch_text_emb in progress_bar:
            batch_graph = batch_graph.to(device)
            batch_text_emb = batch_text_emb.to(device)

            z_graph,_,_ = model(batch_graph)

            loss = contrastive_loss(z_graph, batch_text_emb)
            total_loss += loss.item() * batch_graph.num_graphs

            text_id = retrieve_captioning(z_graph, val_caption_tensor)
            
            pred_caption = [val_data_list[i].description for i in text_id.cpu().numpy()]
            
            individual_graphs = batch_graph.to_data_list()
            true_caption = [graph.description for graph in individual_graphs]
            
            score = evaluator.evaluate_batch(pred_caption, true_caption)
            
            total_score += score['composite_score'] * batch_graph.num_graphs
            total_bleu += score['bleu4_f1_mean'] * batch_graph.num_graphs
            total_bert += score['bertscore_f1_mean'] * batch_graph.num_graphs

            progress_bar.set_postfix(
                v_loss=f'{total_loss / len(dataloader.dataset):.4f}', 
                v_score=f'{total_score / len(dataloader.dataset):.4f}'
            )
    print(f"Validation BLEU-4 F1: {total_bleu / len(dataloader.dataset):.4f}, BERTScore F1: {total_bert / len(dataloader.dataset):.4f}")
    avg_loss = total_loss / len(dataloader.dataset)
    avg_score = total_score / len(dataloader.dataset)
    return avg_loss, avg_score


def main():
    train_data_file = "src/data/train_graphs.pkl"
    val_data_file = "src/data/validation_graphs.pkl"
    train_emb_csv = "src/data/train_embeddings_bge.csv"
    val_emb_csv   = "src/data/validation_embeddings_bge.csv"

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

    train_caption_tensor = embdict_to_tensor(train_emb).to(device)

    train_dataset = PreprocessedGraphDataset(train_data_file, train_emb, encode_feat=True)
    val_dataset = PreprocessedGraphDataset(val_data_file, val_emb, encode_feat=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = GEncoder().to(device)
    #model = GraphT5_GINEncoder().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    from muon import MuonClip, MuonConfig
    muon_config = MuonConfig(enable_clipping=False,
                            lr=learning_rate,
                            muon_decay=weight_decay,
                            log_max_logits=False,
                            log_dir='')
    
    optimizer = MuonClip(model,{}, muon_config)

    total_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.01 * total_steps) 
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    # -------------------------------

    evaluator = MolecularCaptionEvaluator(device=device)

    wandb.watch(model, log="all", log_freq=100)

    best_score = 0.0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, train_caption_tensor, optimizer, scheduler, device, epoch)
        
        wandb.log({
            "train/epoch_loss": train_loss,
            "epoch": epoch
        })
        
        if (epoch) % val_freq == 0:
            val_loss, score = validate_epoch(
                model, 
                val_loader, 
                train_caption_tensor, 
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