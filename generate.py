import torch
import os
from utils.loading_utils import load_model_transformers, load_tokenizer
from utils.data_collection_utils import generate_single_line
torch.set_grad_enabled(False)

INPUT = os.getenv("INPUT")
MODEL = os.getenv("MODEL")
OUTPUT = os.getenv("OUTPUT", f"./experimental_data/{MODEL}/")
FILENAME = os.getenv("FILENAME", "generation.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)

if __name__ == "__main__":
    model = load_model_transformers(MODEL, device=device, dtype=torch.bfloat16)
    tokenizer = load_tokenizer(MODEL, device=device)
    print(model)
    
    output = generate_single_line(model, tokenizer, INPUT, device)
    
    with open(OUTPUT + FILENAME, "w") as f:
        f.write(INPUT + output)