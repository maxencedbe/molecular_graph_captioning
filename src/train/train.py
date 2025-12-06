import torch
import tqdm
import wandb
import os
import torch.optim as optim
from transformers import GPT2Tokenizer

from src.utils import MolecularCaptionEvaluator
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn
from torch.utils.data import DataLoader
from src.model.x_model import MoLCABackbone

epochs = 50
batch_size = 16  # Réduit car GPT-2 est plus lourd
learning_rate = 5e-5  # Plus faible pour éviter catastrophic forgetting
weight_decay = 1e-5
val_freq = 5
save_freq = 10
max_length = 128  # Longueur max des séquences
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, optimizer, tokenizer, device, epoch):
    model.train()
    total_loss = 0
    num_samples_processed = 0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False)
    
    for batch_idx, (batch_graph, batch_text_raw) in enumerate(progress_bar):
        batch_graph = batch_graph.to(device)
        batch_size = batch_graph.num_graphs
        
        # Extraire les descriptions textuelles du batch
        if isinstance(batch_text_raw, dict):
            # Si batch_text_raw est déjà tokenisé
            batch_text = {k: v.to(device) for k, v in batch_text_raw.items()}
        else:
            # Si batch_text_raw contient les descriptions brutes
            descriptions = [data.description for data in batch_graph.to_data_list()]
            
            # Tokeniser les descriptions
            batch_text = tokenizer(
                descriptions,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            batch_text = {k: v.to(device) for k, v in batch_text.items()}
        
        optimizer.zero_grad()

        # Forward pass : calcule automatiquement la cross-entropy loss
        loss = model(batch_graph, batch_text, return_loss=True)
        
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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


def validate_epoch(model, dataloader, val_data_list, evaluator, tokenizer, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    num_samples_processed = 0
    
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_graph, batch_text_raw in progress_bar:
            batch_graph = batch_graph.to(device)
            batch_size = batch_graph.num_graphs
            
            # Extraire les vraies descriptions
            individual_graphs = batch_graph.to_data_list()
            true_captions = [graph.description for graph in individual_graphs]
            
            # Tokeniser pour calcul de loss
            if isinstance(batch_text_raw, dict):
                batch_text = {k: v.to(device) for k, v in batch_text_raw.items()}
            else:
                batch_text = tokenizer(
                    true_captions,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                batch_text = {k: v.to(device) for k, v in batch_text.items()}
            
            # Calcul de la loss
            loss = model(batch_graph, batch_text, return_loss=True)
            total_loss += loss.item() * batch_size
            
            # Génération de prédictions pour évaluation
            # Utiliser un prompt simple
            prompt = "Description: "
            generated_texts = model.generate(batch_graph, prompt, max_length=80)
            
            all_predictions.extend(generated_texts)
            all_references.extend(true_captions)
            
            num_samples_processed += batch_size
            
            progress_bar.set_postfix(
                v_loss=f'{total_loss / num_samples_processed:.4f}'
            )
    
    avg_loss = total_loss / num_samples_processed
    
    # Évaluation avec les métriques
    eval_results = evaluator.evaluate_batch(all_predictions, all_references)
    
    return avg_loss, eval_results


def main():
    train_data_file = "src/data/train_graphs.pkl"
    val_data_file = "src/data/validation_graphs.pkl"

    wandb.init(
        project="molecular-captioning-gpt2",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "architecture": "GEncoder + GPT2",
            "device": str(device)
        }
    )

    print("Loading data and model...")
    
    # Charger le tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Charger les données
    train_data_list = load_data(train_data_file)
    val_data_list = load_data(val_data_file)

    # Dataset personnalisé qui retourne les graphes et les textes
    train_dataset = PreprocessedGraphDataset(train_data_file, encode_feat=True)
    val_dataset = PreprocessedGraphDataset(val_data_file, encode_feat=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # Initialiser le modèle
    model = MoLCABackbone(model_name="gpt2").to(device)
    
    # Optimizer avec différents learning rates
    # Plus faible pour GPT-2, plus élevé pour la projection
    optimizer = optim.AdamW([
        {'params': model.llm.parameters(), 'lr': learning_rate * 0.1},  # Fine-tune léger
        {'params': model.graph_projection.parameters(), 'lr': learning_rate},
        {'params': model.gnn_encoder.parameters(), 'lr': learning_rate}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    evaluator = MolecularCaptionEvaluator(device=device)

    wandb.watch(model, log="all", log_freq=100)

    best_score = 0.0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, tokenizer, device, epoch)
        
        scheduler.step()
        
        wandb.log({
            "train/epoch_loss": train_loss,
            "train/learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })
        
        if (epoch + 1) % val_freq == 0:
            val_loss, eval_results = validate_epoch(
                model, 
                val_loader, 
                val_data_list,
                evaluator,
                tokenizer,
                device
            )
            
            composite_score = eval_results['composite_score']
            
            wandb.log({
                "val/loss": val_loss,
                "val/composite_score": composite_score,
                "val/bleu": eval_results.get('bleu', 0),
                "val/rouge": eval_results.get('rouge', 0),
                "val/meteor": eval_results.get('meteor', 0),
                "epoch": epoch
            })
            
            if composite_score > best_score:
                best_score = composite_score
                os.makedirs("src/saved_model", exist_ok=True)
                torch.save(model.state_dict(), "src/saved_model/best_model_gpt2.pth")
                wandb.save("src/saved_model/best_model_gpt2.pth")
                wandb.run.summary["best_score"] = best_score
                print(f"✓ New best model saved! Score: {best_score:.4f}")
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Score: {composite_score:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
            
        if (epoch+1) % save_freq == 0:
            checkpoint_path = f"src/saved_model/model_gpt2_epoch_{epoch+1}.pth"
            os.makedirs("src/saved_model", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            wandb.save(checkpoint_path)

    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()