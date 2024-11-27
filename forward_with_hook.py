import torch
from utils.data_collection_utils import obtain_acts_diff, obtain_acts_diff_attn, obtain_acts_diff_res
from utils.loading_utils import load_data, load_model
import os
import configparser

DATASET = os.getenv("DATASET")
config = configparser.ConfigParser()
config.read("./data/config.ini")
DATA_PATH = config[DATASET]["data_path"]
MODEL_NAME = os.getenv("MODEL", "gemma-2-2b-it")
PRE_PROMPT = config[DATASET]["pre_prompt"] + " "
PROMPT_KEY = config[DATASET]["prompt_key"]
N_SHOTS = 0
BATCH_SIZE = 1
BATCH = int(os.getenv("BATCH", 0))
OUTPUT = f"./experimental_data/{MODEL_NAME}/{DATASET}/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_IDX = int(os.getenv("START_IDX", 0))
MAX_IDX = int(os.getenv("MAX_IDX", -1))      # Number of indeces in to obtain data, -1 for all
LAYERS = "ALL"    # Layers to look at activations, "ALL" or list of int
STREAM = os.getenv("STREAM")
DIRECT_SAVE = bool(os.getenv("DIRECT_SAVE", False))
torch.set_grad_enabled(False)

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)

if __name__ == "__main__":
    print(f"Executing on device {device}")

    train, test, pair_id_lookup = load_data(N_SHOTS, DATA_PATH, lookup_key=None)

    model = load_model(MODEL_NAME, device=device)
    print(model)
    
    layers = list(range(len(model.blocks))) if LAYERS == "ALL" else LAYERS
    if STREAM == "attn":
        acts_q, acts_q_exp, acts_k, acts_k_exp, acts_v, acts_v_exp = obtain_acts_diff_attn(
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
        acts_q = torch.stack([torch.stack([acts_q[i][layer] for layer in acts_q[i].keys()]) for i in range(len(acts_q))])
        acts_k = torch.stack([torch.stack([acts_k[i][layer] for layer in acts_k[i].keys()]) for i in range(len(acts_k))])
        acts_v = torch.stack([torch.stack([acts_v[i][layer] for layer in acts_v[i].keys()]) for i in range(len(acts_v))])
        acts_q_exp = torch.stack([torch.stack([acts_q_exp[i][layer] for layer in acts_q_exp[i].keys()]) for i in range(len(acts_q_exp))])
        acts_k_exp = torch.stack([torch.stack([acts_k_exp[i][layer] for layer in acts_k_exp[i].keys()]) for i in range(len(acts_k_exp))])
        acts_v_exp = torch.stack([torch.stack([acts_v_exp[i][layer] for layer in acts_v_exp[i].keys()]) for i in range(len(acts_v_exp))])
        acts_q = acts_q[:, :, :, 0, :]
        acts_k = acts_k[:, :, :, 0, :]
        acts_v = acts_v[:, :, :, 0, :]
        acts_q_exp = acts_q_exp[:, :, :, 0, :]
        acts_k_exp = acts_k_exp[:, :, :, 0, :]
        acts_v_exp = acts_v_exp[:, :, :, 0, :]
        torch.save(acts_q, f"{OUTPUT}acts_q_{BATCH}.pt")
        torch.save(acts_k, f"{OUTPUT}acts_k_{BATCH}.pt")
        torch.save(acts_v, f"{OUTPUT}acts_v_{BATCH}.pt")
        torch.save(acts_q_exp, f"{OUTPUT}acts_q_exp_{BATCH}.pt")
        torch.save(acts_k_exp, f"{OUTPUT}acts_k_exp_{BATCH}.pt")
        torch.save(acts_v_exp, f"{OUTPUT}acts_v_exp_{BATCH}.pt")
    elif STREAM == "res":
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
    else:
        acts, acts_exp = obtain_acts_diff(
            model,
            test,
            1,
            " let's think step by step",
            layers,
            PRE_PROMPT,
            STREAM,
            start_idx=START_IDX,
            prompt_key=PROMPT_KEY,
            max_idx=MAX_IDX,
            save_path=OUTPUT + f"{STREAM}/" if DIRECT_SAVE else None
        )
        
        acts = [torch.stack([acts[i][layer] for layer in acts[i].keys()]) for i in range(len(acts))]
        acts_exp = [torch.stack([acts_exp[i][layer] for layer in acts_exp[i].keys()]) for i in range(len(acts_exp))]
        shape_check = True
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                if acts[i].shape != acts[j].shape:
                    shape_check = False
                    break
                    
        if not shape_check:
            for i in range(len(acts)):
                torch.save(acts, f"{OUTPUT}acts_{STREAM}_index_{i}.pt")
                torch.save(acts_exp, f"{OUTPUT}acts_exp_{STREAM}_index_{i}.pt")
        else:
            acts = torch.stack(acts)
            acts_exp = torch.stack(acts_exp)
            torch.save(acts, f"{OUTPUT}acts_{STREAM}_{BATCH}.pt")
            torch.save(acts_exp, f"{OUTPUT}acts_exp_{STREAM}_{BATCH}.pt")
        