import torch
import json
import configparser
import requests
from string import Template
from transformers import AutoTokenizer

CONFIG_PATH = "./config.ini"

def top_k_abs_acts_args(acts: torch.Tensor, k: int, layer: int = -1) -> torch.Tensor:
    """return the top k activations indeces (largest in absolute value) of acts

    Args:
        acts (torch.Tensor): shape (samples, layers, hidden_channels)
        k (int)
        layer (int): the layer to look at, -1 indicates all layers. Defaults to -1.

    Returns:
        torch.Tensor: shape (samples, layers, k)
    """
    acts = abs(acts)
    acts = torch.argsort(acts, dim=-1, descending=True)
    if layer == -1:
        return acts[:, :, :k]
    else:
        return acts[:, layer:layer+1, :k]
    
def fetch_neuron_description(model_name: str, layer: int, neuron_idx: int, stream: str) -> str:
    """fetch neuron description from neuronpedia for specified model, layer and neuron

    Args:
        model_name (str): model name
        layer (int): layer of neuron
        neuron_idx (int): index of neuron
        stream (str): stream of model to look at

    Returns:
        str: description of neuron
    """
    session = requests.Session()
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    url = Template(config[model_name]["neuronpedia_url_template"])
    url = url.substitute(layer=layer, stream=stream, neuron=neuron_idx.item())
    resp = session.get(url)
    if resp.status_code == 200:
        try:
            resp = resp.json()
            desc = resp["explanations"][0]["description"]
            return desc
        except Exception as e:
            print(f"{e}, retrying...")
            return fetch_neuron_description(model_name, layer, neuron_idx, stream)
    else:
        resp.raise_for_status()
        
def checkpoint(path: str, content: object):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)
        
def write_to_file_end(path: str, content: str):
    with open(path, "a") as f:
        f.write(content)

def get_args_desc(
    model_name: str,
    stream: str,
    args: torch.Tensor,
    data_path: str,
    checkpoint_path: str,
    sample_idx: int = 0, 
    layer_idx: int = 0, 
    neuron_idx: int = 0
) -> None:
    """get descriptions for activations, write directly to data_path, write to checkpoint_path when exception
        occurs

    Args:
        model_name (str): name of model
        stream (str): stream to look at
        args (torch.Tensor): shape (samples, layer, num_neurons)
        data_path (str): path to data
        checkpoint_path (str): path to checkpoint file that stores sample_idx, layer_idx and neuron_idx
        sample_idx (int): sample index to pick up from. Defaults to 0
        layer_idx (int): layer index to pick up from. Defaults to 0
        neuron_idx (int): neuron index to pick up from. Defaults to 0
    """
    write_to_file_end(data_path, "[\n")
    while sample_idx < args.shape[0]:
        if sample_idx == 0:
            write_to_file_end(data_path, "    [\n")
        while layer_idx < args.shape[1]:
            if layer_idx == 0:
                write_to_file_end(data_path, "        [\n")
            while neuron_idx < args.shape[2]:
                print(f"sample {sample_idx}, layer {layer_idx}, neuron {neuron_idx}")
                neuron = args[sample_idx, layer_idx, neuron_idx]
                try:
                    neuron_desc = fetch_neuron_description(model_name, layer_idx, neuron, stream)
                    write_to_file_end(data_path, "            \"" + neuron_desc.replace("\"", "\\\"") + ("\", \n" if neuron_idx + 1 != args.shape[2] else "\"\n"))
                except Exception as e:
                    checkpoint(checkpoint_path, {"sample_idx": sample_idx, "layer_idx": layer_idx, "neuron_idx": neuron_idx})
                    raise e
                neuron_idx += 1
            write_to_file_end(data_path, "        ]" + (",\n        [\n" if layer_idx + 1 != args.shape[1] else "\n"))
            neuron_idx = 0
            layer_idx += 1
        write_to_file_end(data_path, "    ]" + (",\n    [\n" if sample_idx + 1 != args.shape[0] else "\n"))
        layer_idx = 0
        sample_idx += 1
    write_to_file_end(data_path, "]")

def get_end_idx(text: str, model_name: str, signal: int, prompt: str):
    """get the end idx in text that signal first appears, in tokenized index

    Args:
        text (str): input text
        model_name (str): name of model to use
        signal (int): signal for stop, token id for stopping word
        prompt (str): prompt given to model that should be removed from text
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    dir = "./models/" + config[model_name]["weights_directory"]
    tokenizer = AutoTokenizer.from_pretrained(dir)
    prompt_len = len(tokenizer(prompt).input_ids)
    text_ids = tokenizer(text).input_ids[prompt_len:]
    index = text_ids.index(signal)
    return index
    