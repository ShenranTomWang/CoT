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
import argparse
import torch
import configparser
from utils.data_collection_utils import get_layer_acts_post_resid, get_layer_acts_attn
from utils.loading_utils import get_pretrained_model

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, help="Input prompt", required=True)
parser.add_argument("-m", "--model", tyoe=str, help="Model name", required=True)
parser.add_argument("-s", "--stream", type=str, help="Stream to take activations, one of 'attn' or 'res'", required=False, default="res")
parser.add_argument("-o", "--output", type=str, help="Output file to write to", required=False, default=None)

config = configparser.ConfigParser()
config = config.read("./config.ini")

if __name__ == "__main__":
    args = parser.parse_args()

    input = args.input
    model = args.model
    stream = args.stream
    output = args.output
    if output == None:
        output = f"./experimental_data/{model}/"
    
    model = config[model]["weights_dir"]
    model = get_pretrained_model(model)
    layers = range(len(model.blocks))
    
    if stream == "res":
        _, acts_resid = get_layer_acts_post_resid([input], model, layers)
        acts_resid = torch.stack([acts_resid[key] for key in acts_resid.keys()], dim=0)
        acts_resid = acts_resid[:, 0, :]
        torch.save(acts_resid, output + "_res.pt")
    elif stream == "attn":
        _, acts_q, acts_k, acts_v = get_layer_acts_attn([input], model, layers)
        acts_q = torch.stack([acts_resid[key] for key in acts_q.keys()], dim=0)
        acts_k = torch.stack([acts_resid[key] for key in acts_k.keys()], dim=0)
        acts_v = torch.stack([acts_resid[key] for key in acts_v.keys()], dim=0)
        acts_q = acts_q[:, 0, :]
        acts_k = acts_k[:, 0, :]
        acts_v = acts_v[:, 0, :]
        torch.save(acts_q, output + "_q.pt")
        torch.save(acts_k, output + "_k.pt")
        torch.save(acts_v, output + "_v.pt")
        