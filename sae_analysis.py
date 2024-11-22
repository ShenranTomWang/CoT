import torch
import os
from pathlib import Path
import json
from utils.analysis_utils import top_k_abs_acts_args, get_args_desc, get_end_idx
from utils.loading_utils import load_dataset

MODEL = os.getenv("MODEL")
DATASET = os.getenv("DATASET")
INDEX = int(os.getenv("INDEX", 0))
DATA_PATH = f"./experimental_data/{MODEL}/{DATASET}/"
SIGNAL = int(os.getenv("SIGNAL", -1))
SIGNAL_EXP = int(os.getenv("SIGNAL_EXP", None))
STREAM = os.getenv("STREAM", "res")
TOP_K = int(os.getenv("TOP_K", 1))
LAYER = int(os.getenv("LAYER", -1))
CACHE_PATH = "./cache/ckpt.json"
CACHE_EXP_PATH = "./cache/ckpt_exp.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, data, _, prompt_key, pre_prompt = load_dataset(0, DATASET)

if SIGNAL != -1:
    with open(DATA_PATH + f"generation_{INDEX}.txt", "r") as f:
        text = f.read()
    chop_off = get_end_idx(text, MODEL, SIGNAL, pre_prompt + data[prompt_key][INDEX])

if SIGNAL_EXP != -1:
    with open(DATA_PATH + f"generation_exp_{INDEX}.txt", "r") as f:
        text = f.read()
    chop_off_exp = get_end_idx(text, MODEL, SIGNAL_EXP, pre_prompt + data[prompt_key][INDEX] + " Let's think step by step.")

acts_resid = torch.load(DATA_PATH + f"acts_resid_generation_{INDEX}.pt", map_location=device)
acts_resid = acts_resid[:chop_off, :, :]
acts_exp_resid = torch.load(DATA_PATH + f"acts_exp_resid_generation_{INDEX}.pt", map_location=device)
acts_exp_resid = acts_exp_resid[:chop_off_exp, :, :]
print(f"acts_resid.shape: {acts_resid.shape}, acts_exp_resid.shape: {acts_exp_resid.shape}")

acts_resid_args = top_k_abs_acts_args(acts_resid, TOP_K, layer=LAYER)      # (samples, layers, TOP_K)
acts_exp_resid_args = top_k_abs_acts_args(acts_exp_resid, TOP_K, layer=LAYER)

# if Path(CACHE_PATH).exists():
#     with open(CACHE_PATH, "r") as cache:
#         obj = json.load(cache)
#     resid_desc = get_args_desc(
#         MODEL, STREAM, acts_resid_args, DATA_PATH + f"acts_desc_{INDEX}.json", CACHE_PATH,
#         sample_idx=obj.sample_idx, layer_idx=obj.layer_idx, neuron_idx=obj.neuron_idx
#     )
# else:
#     os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
#     resid_desc = get_args_desc(
#         MODEL, STREAM, acts_resid_args, DATA_PATH + f"acts_desc_exp_{INDEX}.json", CACHE_PATH
#     )

if Path(CACHE_EXP_PATH).exists():
    with open(CACHE_EXP_PATH, "r") as cache:
        obj = json.load(cache)
    resid_exp_desc = get_args_desc(
        MODEL, STREAM, acts_resid_args, DATA_PATH + f"acts_desc_{INDEX}.json", CACHE_EXP_PATH,
        sample_idx=obj.sample_idx, layer_idx=obj.layer_idx, neuron_idx=obj.neuron_idx
    )
else:
    os.makedirs(os.path.dirname(CACHE_EXP_PATH), exist_ok=True)
    resid_exp_desc = get_args_desc(
        MODEL, STREAM, acts_exp_resid_args, DATA_PATH + f"acts_exp_desc_{INDEX}.json", CACHE_EXP_PATH
    )
