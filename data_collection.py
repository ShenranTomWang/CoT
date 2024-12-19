import torch
import os
from utils.loading_utils import load_model_transformers, load_tokenizer
from utils.data_collection_utils import generate_single_line
import data.dataset
Dataset = getattr(data.dataset, os.getenv("DATASET"))
torch.set_grad_enabled(False)
MODEL = os.getenv("MODEL")
MAX_IDX = int(os.getenv("MAX_IDX", -1))
OUTPUT = os.getenv("OUTPUT", f"./experimental_data/{MODEL}/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load()
    model = load_model_transformers(MODEL, device=device, dtype=torch.float16, trust_remote_code=True)
    tokenizer = load_tokenizer(MODEL, device=device, trust_remote_code=True)
    print(model)
    
    max_idx = MAX_IDX if MAX_IDX != -1 else len(dataset)
    for i in range(max_idx):
        question = dataset[i][dataset.prompt_key]
        question = question + " Format your response such that you show your reasoning first and end with \"Answer: (your numerical answer here)\""
        output = generate_single_line(model, tokenizer, question, device, max_len=100)
        dataset[i][MODEL] = output
    
    dataset.save_as(path=OUTPUT)