import torch
import os
import json
from utils.analysis_utils import top_k_abs_acts_args, get_args_desc

MODEL = os.getenv("MODEL")
DATASET = os.getenv("DATASET")
INDEX = int(os.getenv("INDEX"))
DATA_PATH = f"./experimental_data/{MODEL}/{DATASET}/"
CHOP_OFF = int(os.getenv("CHOP_OFF"))
STREAM = os.getenv("STREAM", "res")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acts_resid = torch.load(DATA_PATH + f"acts_resid_generation_{INDEX}.pt", map_location=device)
acts_resid = acts_resid[:CHOP_OFF, :, :]
acts_exp_resid = torch.load(DATA_PATH + f"acts_exp_resid_generation_{INDEX}.pt", map_location=device)

acts_resid_args = top_k_abs_acts_args(acts_resid, 1)      # (samples, layers, 1)
acts_exp_resid_args = top_k_abs_acts_args(acts_exp_resid, 1)

resid_desc = get_args_desc(MODEL, STREAM, acts_resid_args)
resid_exp_desc = get_args_desc(MODEL, STREAM, acts_exp_resid_args)

with open(DATA_PATH + "acts_exp_desc.json", "w") as f:
    json.dump(resid_exp_desc, f, indent=4)
    
with open(DATA_PATH + "acts_desc.json", "w") as f:
    json.dump(resid_desc, f, indent=4)
