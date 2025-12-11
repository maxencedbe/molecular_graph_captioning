import torch
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn
from torch.utils.data import DataLoader
from src.model.model_T5 import GraphCapT5
from src.utils import MolecularCaptionEvaluator
from transformers import T5Tokenizer

def print_separator():
    print("\n" + "="*100 + "\n")

def load_model_and_data(checkpoint_path, val_data_file, device):
    """Load model, tokenizer and validation data"""
    print("Loading model and data...")
    
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-base", model_max_length=512)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load validation data
    val_data_list = load_data(val_data_file)
    val_dataset = PreprocessedGraphDataset(val_data_file, encode_feat=True)
    
    # Create dataloader with batch_size=1 for easier inspection
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    model = GraphCapT5(
        model_name="GT4SD/multitask-text-and-chemistry-t5-base-standard",
        freeze_encoder=True,
        freeze_decoder=False,
        freeze_graph_encoder=True,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded checkpoint (state dict only)")
    
    model.eval()
    print(f"✓ Model loaded on {device}")
    print(f"✓ Validation set size: {len(val_dataset)} samples\n")
    
    return model, tokenizer, val_loader, val_data_list

def generate_predictions(model, val_loader, val_data_list, num_examples=10, specific_indices=None):
    """Generate predictions for validation samples"""
    
    device = next(model.parameters()).device
    all_results = []
    
    print(f"Generating predictions for {num_examples} examples...")
    print_separator()
    
    with torch.no_grad():
        for idx, (batch_graph, batch_smiles, batch_descriptions) in enumerate(val_loader):
            # Skip if we're only doing specific indices
            if specific_indices is not None and idx not in specific_indices:
                continue
            
            # Stop if we have enough examples
            if len(all_results) >= num_examples:
                break
            
            batch_graph = batch_graph.to(device)
            
            # Generate prediction
            generated_text = model.generate(
                batch_graph, 
                batch_smiles, 
                max_length=512, 
                num_beams=5
            )[0]  # Get first (and only) element since batch_size=1
            
            # Get ground truth
            ground_truth = batch_descriptions[0]
            smiles = batch_smiles[0] if isinstance(batch_smiles, list) else batch_smiles
            
            # Store result
            result = {
                'index': idx,
                'smiles': smiles,
                'ground_truth': ground_truth,
                'prediction': generated_text
            }
            all_results.append(result)
            
            # Print result
            print(f"📌 Example {len(all_results)} (Index: {idx})")
            print(f"SMILES: {smiles}")
            print(f"\n🎯 Ground Truth:")
            print(f"   {ground_truth}")
            print(f"\n🤖 Model Prediction:")
            print(f"   {generated_text}")
            print_separator()
    
    return all_results

def analyze_predictions(results, device='cuda'):
    """Analyze prediction quality with scores"""
    print("\n📊 ANALYSIS SUMMARY")
    print_separator()
    
    # Length statistics
    gt_lengths = [len(r['ground_truth'].split()) for r in results]
    pred_lengths = [len(r['prediction'].split()) for r in results]
    
    print(f"Ground Truth - Avg length: {sum(gt_lengths)/len(gt_lengths):.1f} words")
    print(f"Predictions  - Avg length: {sum(pred_lengths)/len(pred_lengths):.1f} words")
    print()
    
    # Check for repetitions or empty predictions
    empty_preds = sum(1 for r in results if len(r['prediction'].strip()) == 0)
    print(f"Empty predictions: {empty_preds}/{len(results)}")
    
    # Check for diversity
    unique_preds = len(set(r['prediction'] for r in results))
    print(f"Unique predictions: {unique_preds}/{len(results)}")
    print_separator()
    
    # Calculate evaluation metrics
    print("🎯 CALCULATING EVALUATION METRICS...")
    evaluator = MolecularCaptionEvaluator(device=device)
    
    predictions = [r['prediction'] for r in results]
    references = [r['ground_truth'] for r in results]
    
    eval_results = evaluator.evaluate_batch(predictions, references)
    
    print("\n📈 SCORES:")
    print(f"   Composite Score:  {eval_results['composite_score']:.4f}")
    print(f"   BLEU-4 F1:        {eval_results.get('bleu4_f1_mean', 0):.4f}")
    print(f"   BERTScore F1:     {eval_results.get('bertscore_f1_mean', 0):.4f}")
    
    # Show BLEU scores breakdown if available
    if 'bleu1_f1_mean' in eval_results:
        print(f"\n   BLEU-1 F1:        {eval_results.get('bleu1_f1_mean', 0):.4f}")
        print(f"   BLEU-2 F1:        {eval_results.get('bleu2_f1_mean', 0):.4f}")
        print(f"   BLEU-3 F1:        {eval_results.get('bleu3_f1_mean', 0):.4f}")
    
    print()
    return eval_results

def main():
    # Configuration
    checkpoint_path = "src/saved_model/best_model_t5_aligned.pth"  # Change this to your checkpoint
    val_data_file = "src/data/validation_graphs_smiles.pkl"
    num_examples = 10  # Number of examples to show
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # You can also specify specific indices to check
    # specific_indices = [0, 5, 10, 42, 100]  # Uncomment to use specific samples
    specific_indices = None
    
    print("🚀 Molecular Caption Inference")
    print_separator()
    
    # Load model and data
    model, tokenizer, val_loader, val_data_list = load_model_and_data(
        checkpoint_path, 
        val_data_file, 
        device
    )
    
    # Generate predictions
    results = generate_predictions(
        model, 
        val_loader, 
        val_data_list, 
        num_examples=num_examples,
        specific_indices=specific_indices
    )
    
    # Analyze results and calculate scores
    eval_results = analyze_predictions(results, device=device)
    
    print("\n✅ Inference completed!")

if __name__ == "__main__":
    main()