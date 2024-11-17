import torch
from utils.data_collection_utils import load_data, load_model, obtain_single_line_generation_act, load_tokenizer
import os
import configparser

DATASET = os.getenv("DATASET")
config = configparser.ConfigParser()
config.read("./data/config.ini")
DATA_PATH = config[DATASET]["data_path"]
MODEL_NAME = "gemma-2-2b"
PRE_PROMPT = config[DATASET]["pre_prompt"] + " "
PROMPT_KEY = config[DATASET]["prompt_key"]
N_SHOTS = 0
INDEX = int(os.getenv("BATCH"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_IDX = int(os.getenv("START_IDX"))
MAX_IDX = int(os.getenv("MAX_IDX"))      # Number of indeces in to obtain data, -1 for all
LAYERS = "ALL"    # Layers to look at activations, "ALL" or list of int
torch.set_grad_enabled(False)

if __name__ == "__main__":
    print(f"Executing on device {device}")

    train, test, pair_id_lookup = load_data(N_SHOTS, DATA_PATH, lookup_key=None)

    model = load_model(MODEL_NAME, device=device)
    print(model)
    
    tokenizer = load_tokenizer(MODEL_NAME, device=device)
    
    layers = list(range(len(model.blocks))) if LAYERS == "ALL" else LAYERS
    acts_resid, acts_resid_exp, _, _ = obtain_single_line_generation_act(
        model,
        test.iloc[INDEX]["sent"],
        " Let's think step by step",
        LAYERS,
        PRE_PROMPT,
        tokenizer
    )
    
    acts_resid = torch.stack([torch.stack([acts_resid[i][layer] for layer in acts_resid[i].keys()]) for i in range(len(acts_resid))])
    acts_exp_resid = torch.stack([torch.stack([acts_resid_exp[i][layer] for layer in acts_resid_exp[i].keys()]) for i in range(len(acts_resid_exp))])
    acts_resid = acts_resid[:, :, 0, :]
    acts_exp_resid = acts_exp_resid[:, :, 0, :]
    torch.save(acts_resid, f"./experimental_data/{MODEL_NAME}/{DATASET}/acts_resid_generation_{INDEX}.pt")
    torch.save(acts_exp_resid, f"./experimental_data/{MODEL_NAME}/{DATASET}/acts_exp_resid_generation_{INDEX}.pt")