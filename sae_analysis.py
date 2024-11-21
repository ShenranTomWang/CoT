import torch
import os
import json
from utils.analysis_utils import top_k_abs_acts_args, get_args_desc

MODEL = os.getenv("MODEL")
DATASET = os.getenv("DATASET")
INDEX = int(os.getenv("INDEX", 0))
DATA_PATH = f"./experimental_data/{MODEL}/{DATASET}/"
CHOP_OFF = int(os.getenv("CHOP_OFF", -1))
CHOP_OFF_EXP = int(os.getenv("CHOP_OFF_EXP", -1))
STREAM = os.getenv("STREAM", "res")
TOP_K = int(os.getenv("TOP_K", 1))
LAYER = int(os.getenv("LAYER", -1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

acts_resid = torch.load(DATA_PATH + f"acts_resid_generation_{INDEX}.pt", map_location=device)
acts_resid = acts_resid[:CHOP_OFF, :, :]
acts_exp_resid = torch.load(DATA_PATH + f"acts_exp_resid_generation_{INDEX}.pt", map_location=device)
acts_exp_resid = acts_exp_resid[:CHOP_OFF_EXP, :, :]

acts_resid_args = top_k_abs_acts_args(acts_resid, TOP_K, layer=LAYER)      # (samples, layers, TOP_K)
acts_exp_resid_args = top_k_abs_acts_args(acts_exp_resid, TOP_K, layer=LAYER)

resid_desc = get_args_desc(MODEL, STREAM, acts_resid_args)
with open(DATA_PATH + f"acts_desc_{INDEX}.json", "w") as f:
    json.dump(resid_desc, f, indent=4)

resid_exp_desc = get_args_desc(MODEL, STREAM, acts_exp_resid_args)
with open(DATA_PATH + f"acts_exp_desc_{INDEX}.json", "w") as f:
    json.dump(resid_exp_desc, f, indent=4)
