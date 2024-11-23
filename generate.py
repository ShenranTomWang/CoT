"""
This script generates and saves activations from a specified model based on the input prompt.
Usage:
    python generate.py -i <input_prompt> -m <model_name> [-s <stream>] [-o <output_file>]
Arguments:
    -i, --input (str): Input prompt to generate activations for. (required)
    -m, --model (str): Name of the model to use for generating activations. (required)
    -s, --stream (str): Stream to take activations from, one of 'attn' or 'res'. Defaults to 'res'. (optional)
    -o, --output (str): Output file to write activations to. Defaults to "./experimental_data/<model_name>/". (optional)
Configuration:
    The script reads model weights directory from a configuration file named "config.ini".
Output:
    Saves the activations to the specified output file(s) in PyTorch tensor format.
    - For 'res' stream: Saves activations to "<output_file>_res.pt".
    - For 'attn' stream: Saves activations to "<output_file>_q.pt", "<output_file>_k.pt", and "<output_file>_v.pt".
"""
import torch
import os
from utils.data_collection_utils import get_layer_acts_post_resid, get_layer_acts_attn
from utils.loading_utils import load_model

INPUT = os.getenv("INPUT")
MODEL = os.getenv("MODEL")
STREAM = os.getenv("STREAM")
OUTPUT = os.getenv("OUTPUT", f"./experimental_data/{MODEL}/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = load_model(MODEL, device, dtype=torch.bfloat16)
    layers = range(len(model.blocks))
    
    if STREAM == "res":
        _, acts_resid = get_layer_acts_post_resid([input], model, layers)
        acts_resid = torch.stack([acts_resid[key] for key in acts_resid.keys()], dim=0)
        acts_resid = acts_resid[:, 0, :]
        torch.save(acts_resid, OUTPUT + "acts_res.pt")
    elif STREAM == "attn":
        _, acts_q, acts_k, acts_v = get_layer_acts_attn([input], model, layers)
        acts_q = torch.stack([acts_q[key] for key in acts_q.keys()], dim=0)
        acts_k = torch.stack([acts_k[key] for key in acts_k.keys()], dim=0)
        acts_v = torch.stack([acts_v[key] for key in acts_v.keys()], dim=0)
        acts_q = acts_q[:, 0, :]
        acts_k = acts_k[:, 0, :]
        acts_v = acts_v[:, 0, :]
        torch.save(acts_q, OUTPUT + "acts_q.pt")
        torch.save(acts_k, OUTPUT + "acts_k.pt")
        torch.save(acts_v, OUTPUT + "acts_v.pt")
        