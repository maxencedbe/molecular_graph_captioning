import torch
import tqdm
import wandb
import os
import torch.optim as optim
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from src.model.model_chemlm import GraphChemLLM
from src.data.data_process import load_data
from src.utils import MolecularCaptionEvaluator
from torch_geometric.data import Batch
from torch.optim import AdamW

epochs = 1
batch_size = 4
learning_rate = 5e-5
weight_decay = 1e-5
val_freq = 5
save_freq = 10
gradient_accumulation_steps = 4
prompt = "\n Describe this molecule:"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphTextDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, tokenizer, prompt_text=prompt, max_length=512):
        self.graphs = graphs
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.max_length = max_length

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        text = graph.description 
        text = self.prompt_text + " " + text
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return graph, tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)

def collate_fn_llm(batch):

    graphs, input_ids, masks = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(input_ids), torch.stack(masks)

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    optimizer.zero_grad()

    for batch_idx, (batch_graph, input_ids, attention_mask) in enumerate(progress_bar):
        batch_graph = batch_graph.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(
            batch_graph=batch_graph,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids 
        )

        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * gradient_accumulation_steps

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        wandb.log({
            "train/batch_loss": loss.item() * gradient_accumulation_steps,
            "train/learning_rate": scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr'],
            "train/step": epoch * len(dataloader) + batch_idx
        })

        if batch_idx % 1000 == 0:
            checkpoint_path = f"src/saved_model/model_step_{batch_idx}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, evaluator, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm.tqdm(dataloader, desc="Validation LLM", leave=False)
    
    with torch.no_grad():
        for batch_graph, input_ids, attention_mask in progress_bar:
            batch_graph = batch_graph.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                batch_graph=batch_graph,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            total_loss += outputs.loss.item() * batch_graph.num_graphs


            prompt_text = [prompt] * batch_graph.num_graphs
            

            generated_captions = model.generate(
                batch_graph=batch_graph, 
                prompt_text=prompt_text, 
                max_new_tokens=512
            )

            true_captions = model.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            all_predictions.extend(generated_captions)
            all_targets.extend(true_captions)

            progress_bar.set_postfix(v_loss=f'{total_loss / len(dataloader.dataset):.4f}')

    scores = evaluator.evaluate_batch(all_predictions, all_targets)
    
    print(f"\n--- Validation Results ---")
    print(f"BLEU-4 F1: {scores.get('bleu4', 0):.4f}")
    print(f"BERTScore F1: {scores.get('bertscore', 0):.4f}")
    print(f"Composite: {scores.get('composite_score', 0):.4f}")
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, scores['composite_score']

def main():

    wandb.init(
        project="molecular-captioning-chemlm",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "architecture": "GEncoder+LLM",
            "device": str(device)
        }
    )

    model = GraphChemLLM(
        llm_model_id="AI4Chem/ChemLLM-2b-1_5",
        freeze_llm=True
    ).to(device)

    model.load_state_dict(torch.load("src/saved_model/best_model_gencoder.pth", map_location=device), strict=False)

    train_data = load_data("src/data/train_graphs.pkl")
    train_dataset = GraphTextDataset(train_data, model.tokenizer)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_llm
    )

    val_data = load_data("src/data/validation_graphs.pkl")
    val_dataset = GraphTextDataset(val_data, model.tokenizer)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_llm
    )


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-5)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    for epoch in range(epochs):
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        wandb.log({
            "train/epoch_loss": avg_loss,
            "epoch": epoch
        })
        
        if epoch % val_freq == 0:
            evaluator = MolecularCaptionEvaluator()

            val_loss, val_score = validate_epoch(model, val_loader, evaluator, device)

            wandb.log({
                "val/loss": val_loss,
                "val/score": val_score,
                "epoch": epoch
            })
            

            if epoch % save_freq == 0:
                torch.save(model.state_dict(), f"graph_llm_epoch_{epoch}_valscore_{val_score:.4f}.pth")

    wandb.finish()


if __name__ == "__main__":
    main()