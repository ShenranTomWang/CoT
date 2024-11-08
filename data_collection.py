import torch
from utils.data_collection_utils import load_data, load_model, obtain_act_diff
import os

DATASET = "com2sense"
DATA_PATH = f"./data/{DATASET}.json"
MODEL_NAME = "gemma-2-2b-it"
PRE_PROMPT = "Yes or no: "
PROMPT_KEY = "input"
N_SHOTS = 0
BATCH_SIZE = 1
BATCH = int(os.getenv("BATCH"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
remote = False
START_IDX = int(os.getenv("START_IDX"))
MAX_IDX = int(os.getenv("MAX_IDX"))      # Number of indeces in to obtain data, -1 for all
LAYERS = "ALL"    # Layers to look at activations, "ALL" or list of int
torch.set_grad_enabled(False)

if __name__ == "__main__":
    print(f"Executing on device {device}")

    train, test, pair_id_lookup = load_data(N_SHOTS, DATA_PATH)

    model = load_model(MODEL_NAME, device=device)
    print(model)
    
    layers = list(range(len(model.blocks))) if LAYERS == "ALL" else LAYERS
    diffs_resid, acts_resid, acts_resid_exp = obtain_act_diff(
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
    torch.save(diffs_resid, f"./experimental_data/{MODEL_NAME}/{DATASET}/diffs_resid_{BATCH}.pt")
    torch.save(acts_resid, f"./experimental_data/{MODEL_NAME}/{DATASET}/acts_resid_{BATCH}.pt")
    torch.save(acts_exp_resid, f"./experimental_data/{MODEL_NAME}/{DATASET}/acts_exp_resid_{BATCH}.pt")