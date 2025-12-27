import torch
import tqdm
import wandb
import os
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.utils import MolecularCaptionEvaluator

epochs = 1
batch_size = 1
learning_rate = 1e-6
weight_decay = 1e-5
val_freq = 2
save_freq = 5
max_length = 512

num_generations = 2
beta = 0.05
epsilon = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import Dict, List, Tuple

class PreprocessedGraphDataset(Dataset):
    def __init__(self, graph_path: str, 
                 emb_dict: Dict[str, torch.Tensor] = None,
                 encode_feat: bool = True):
        print(f"Loading graphs from: {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graphs = pickle.load(f)
            
        self.emb_dict = emb_dict

        self.ids = [getattr(g, 'id', i) for i, g in enumerate(self.graphs)]
        self.encode_feat = encode_feat
        print(f"Loaded {len(self.graphs)} graphs")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]        
        smiles = getattr(graph, 'smiles', "")
        description = getattr(graph, 'description', "")
        
        if self.encode_feat: 

            pass

        return graph, smiles, description


def collate_fn(batch):

    graphs, smiles_list, description_list = zip(*batch)    
    batch_graph = Batch.from_data_list(list(graphs))
    return batch_graph, list(smiles_list), list(description_list)




class TextRLOOTrainer:
    def __init__(self, model, ref_model, evaluator, tokenizer, device):
        self.model = model
        self.ref_model = ref_model
        self.evaluator = evaluator
        self.tokenizer = tokenizer
        self.device = device
        
        self.num_generations = num_generations
        self.beta = beta
        self.epsilon = epsilon
        
    def generate_samples(self, batch_smiles):
        self.model.eval()
        
        all_completions = []
        all_completion_ids = []
        all_old_logps = []
        all_ref_logps = []
        
        input_texts = [f"Caption the following SMILES: {s.strip()}" for s in batch_smiles]
        inputs = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            for _ in range(self.num_generations):
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7, 
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                
                generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                completion_tokens = self.tokenizer(
                    generated_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                old_logps = self._compute_logps(
                    self.model,
                    inputs,
                    completion_tokens
                )
                all_old_logps.append(old_logps)
                
                if self.ref_model:
                    ref_logps = self._compute_logps(
                        self.ref_model,
                        inputs,
                        completion_tokens
                    )
                    all_ref_logps.append(ref_logps)
                
                all_completions.append(generated_texts)
                all_completion_ids.append(completion_tokens)
        
        return {
            "completions": all_completions,
            "completion_ids": all_completion_ids,
            "old_logps": torch.stack(all_old_logps, dim=1),
            "ref_logps": torch.stack(all_ref_logps, dim=1) if all_ref_logps else None
        }
    
    def _compute_logps(self, model, inputs, target_tokens):
        target_ids = target_tokens['input_ids']
        
        # T5 hack: prepare decoder_input_ids explicitly if needed, but T5ForConditionalGeneration
        # handles shifting internally if we pass 'labels'. However, to get accurate log_probs aligned 
        # with logits, passing labels is the standard way.
        
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=target_ids,
            return_dict=True
        )
        
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        mask = target_tokens['attention_mask'].float()
        labels_mask = (target_ids != self.tokenizer.pad_token_id).float()
        final_mask = mask * labels_mask

        sequence_log_probs = (selected_log_probs * final_mask).sum(dim=-1)
        
        return sequence_log_probs
    
    def compute_rewards(self, all_completions, all_references):
        all_rewards = []
        
        for gen_idx in range(self.num_generations):
            completions = all_completions[gen_idx]
            rewards = []
            for pred, ref in zip(completions, all_references):
                eval_result = self.evaluator.evaluate_batch([pred], [ref])
                reward = eval_result.get('bleu4_f1_mean', 0.0) 
                
                # Penalty for chemical nonsense if needed
                if "=" in pred and len(pred.split()) < 3:
                     reward -= 0.5

                rewards.append(reward)

            rewards_tensor = torch.tensor(rewards, device=self.device)
            all_rewards.append(rewards_tensor)
        
        return torch.stack(all_rewards, dim=1)
    
    def compute_loss(self, batch_smiles, batch_descriptions):
        samples = self.generate_samples(batch_smiles)
        
        rewards = self.compute_rewards(
            samples["completions"],
            batch_descriptions
        )
        
        if self.ref_model and samples["ref_logps"] is not None:
            log_ratio_ref = samples["old_logps"] - samples["ref_logps"]
            kl = torch.exp(log_ratio_ref) - 1 - log_ratio_ref
            rewards = rewards - self.beta * kl
            mean_kl = kl.mean().item()
        else:
            mean_kl = 0.0
        
        sum_rewards = rewards.sum(dim=1, keepdim=True)
        baselines = (sum_rewards - rewards) / (self.num_generations - 1)
        advantages = (rewards - baselines) 

        self.model.train()
        all_new_logps = []
        
        input_texts = [f"Caption the following SMILES: {s.strip()}" for s in batch_smiles]
        inputs = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(self.device)
        
        for gen_idx in range(self.num_generations):
            completion_tokens = samples["completion_ids"][gen_idx]
            new_logps = self._compute_logps(
                self.model,
                inputs,
                completion_tokens
            )
            all_new_logps.append(new_logps)
        
        new_logps = torch.stack(all_new_logps, dim=1).view(-1)
        old_logps = samples["old_logps"].view(-1)
        advantages = advantages.view(-1)
        
        log_ratio = new_logps - old_logps
        log_ratio = torch.clamp(log_ratio, min=-10, max=10) 
        ratio = torch.exp(log_ratio)
        
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages

        loss = -torch.min(loss1, loss2).mean()
        
        return {
            "loss": loss,
            "reward": rewards.mean().item(),
            "kl": mean_kl
        }

def train_epoch_rloo(trainer, dataloader, optimizer, device, epoch):
    total_loss = 0
    total_reward = 0
    total_kl = 0
    num_batches = 0
    
    progress_bar = tqdm.tqdm(dataloader, desc=f"RLOO Epoch {epoch+1}", leave=False)
    
    for batch_idx, (batch_graph, batch_smiles, batch_descriptions) in enumerate(progress_bar):
        
        optimizer.zero_grad()
        
        try:
            metrics = trainer.compute_loss(batch_smiles, batch_descriptions)
            loss = metrics["loss"]
            
            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_reward += metrics["reward"]
            total_kl += metrics["kl"]
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'r': f'{metrics["reward"]:.4f}',
                'kl': f'{metrics["kl"]:.4f}'
            })
            
            global_step = epoch * len(dataloader) + batch_idx
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_reward": metrics["reward"],
                "train/batch_kl": metrics["kl"],
                "train/step": global_step
            })

        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    if num_batches == 0: return {"loss": 0.0, "reward": 0.0, "kl": 0.0}
    
    return {
        "loss": total_loss / num_batches,
        "reward": total_reward / num_batches,
        "kl": total_kl / num_batches
    }

def validate_epoch_rloo(trainer, dataloader, device):
    trainer.model.eval()
    total_reward = 0
    all_predictions = []
    all_references = []
    num_samples = 0
    
    progress_bar = tqdm.tqdm(dataloader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_graph, batch_smiles, batch_descriptions in progress_bar:
            
            input_texts = [f"Caption the following SMILES: {s.strip()}" for s in batch_smiles]
            inputs = trainer.tokenizer(
                input_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            ).to(device)
            
            generated_ids = trainer.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                repetition_penalty=1.5,
                do_sample=False
            )
            
            generated_texts = trainer.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_predictions.extend(generated_texts)
            all_references.extend(batch_descriptions)
            
            # Simple average reward for logging
            for pred, ref in zip(generated_texts, batch_descriptions):
                eval_result = trainer.evaluator.evaluate_batch([pred], [ref])
                total_reward += eval_result.get('bleu4_f1_mean', 0.0)
            
            num_samples += len(generated_texts)
    
    avg_reward = total_reward / num_samples if num_samples > 0 else 0
    eval_results = trainer.evaluator.evaluate_batch(all_predictions, all_references)
    
    return avg_reward, eval_results

def main():
    train_data_file = "src/data/train_graphs_selfies.pkl"
    val_data_file = "src/data/validation_graphs_selfies.pkl"
    model_name = "GT4SD/multitask-text-and-chemistry-t5-base-augm"

    wandb.init(project="t5-chemistry-rloo")

    train_dataset = PreprocessedGraphDataset(train_data_file, encode_feat=True)
    val_dataset = PreprocessedGraphDataset(val_data_file, encode_feat=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(val_dataset, range(100)), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    evaluator = MolecularCaptionEvaluator(device=device)

    trainer = TextRLOOTrainer(
        model=model,
        ref_model=ref_model,
        evaluator=evaluator,
        tokenizer=tokenizer,
        device=device
    )
    
    best_score = 0.0
    
    for epoch in range(epochs):
        train_metrics = train_epoch_rloo(trainer, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Loss: {train_metrics['loss']:.4f} | R: {train_metrics['reward']:.4f}")
        
        if (epoch + 1) % val_freq == 0:
            val_reward, eval_results = validate_epoch_rloo(trainer, val_loader, device)
            composite_score = eval_results.get('bleu4_f1_mean', 0.0)
            
            print(f"Validation Score: {composite_score:.4f}")
            if composite_score > best_score:
                best_score = composite_score
                torch.save(model.state_dict(), "best_model_rloo_t5.pth")

    wandb.finish()

if __name__ == "__main__":
    main()