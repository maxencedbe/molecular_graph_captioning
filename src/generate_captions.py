from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from src.data.data_process import load_data, PreprocessedGraphDataset, collate_fn, load_id2emb, embdict_to_tensor
from src.utils import  MolecularCaptionEvaluator

custom_cache_dir = "./cache_model/"
model_name = "GT4SD/multitask-text-and-chemistry-t5-base-augm"

tokenizer = T5Tokenizer.from_pretrained(
    model_name, 
    cache_dir=custom_cache_dir
)

model = T5ForConditionalGeneration.from_pretrained(
    model_name, 
    cache_dir=custom_cache_dir
)

val_data_file = "src/data/validation_graphs.pkl"
val_data = load_data(val_data_file)
evaluator = MolecularCaptionEvaluator()

mean = 0.0

for i,data in enumerate(val_data):
    input_text = f"Caption the following : {data.description}"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(
        **inputs, 
        max_length=512,
        num_beams=5
    )
    
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"SMILES: {data.description}\nDescription: {description}\n")
    score = evaluator.evaluate_batch([description],[data.description])['bleu4_f1_mean']
    mean += score
    print(f"meannnnnnnnnn : {mean/(i+1)}")
    print(evaluator.evaluate_batch([description],[data.description])['bleu4_f1_mean'])
