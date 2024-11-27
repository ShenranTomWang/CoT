import torch
from utils.data_collection_utils import obtain_acts_diff_res
from utils.loading_utils import load_data, load_model
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
BATCH_SIZE = 1
BATCH = int(os.getenv("BATCH"))
OUTPUT = "./experimental_data/{MODEL_NAME}/{DATASET}/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_IDX = int(os.getenv("START_IDX"))
MAX_IDX = int(os.getenv("MAX_IDX"))      # Number of indeces in to obtain data, -1 for all
LAYERS = "ALL"    # Layers to look at activations, "ALL" or list of int
torch.set_grad_enabled(False)

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)

if __name__ == "__main__":
    print(f"Executing on device {device}")

    train, test, pair_id_lookup = load_data(N_SHOTS, DATA_PATH, lookup_key=None)

    model = load_model(MODEL_NAME, device=device)
    print(model)
    
    layers = list(range(len(model.blocks))) if LAYERS == "ALL" else LAYERS
    diffs_resid, acts_resid, acts_resid_exp = obtain_acts_diff_res(
        model,
        test,
        1,
        " let's think step by step",
        layers,
        PRE_PROMPT,
        start_idx=START_IDX,
        prompt_key=PROMPT_KEY,
        max_idx=MAX_IDX
    )
    
    diffs_resid = torch.stack([torch.stack([diffs_resid[i][layer] for layer in diffs_resid[i].keys()]) for i in range(len(diffs_resid))])
    acts_resid = torch.stack([torch.stack([acts_resid[i][layer] for layer in acts_resid[i].keys()]) for i in range(len(acts_resid))])
    acts_exp_resid = torch.stack([torch.stack([acts_resid_exp[i][layer] for layer in acts_resid_exp[i].keys()]) for i in range(len(acts_resid_exp))])
    diffs_resid = diffs_resid[:, :, 0, :]
    acts_resid = acts_resid[:, :, 0, :]
    acts_exp_resid = acts_exp_resid[:, :, 0, :]
    torch.save(diffs_resid, f"{OUTPUT}diffs_resid_{BATCH}.pt")
    torch.save(acts_resid, f"{OUTPUT}acts_resid_{BATCH}.pt")
    torch.save(acts_exp_resid, f"{OUTPUT}acts_exp_resid_{BATCH}.pt")