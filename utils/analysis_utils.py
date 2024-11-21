import torch
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

def get_args_desc(model_name: str, stream: str, args: torch.Tensor) -> list:
    """get descriptions for activations

    Args:
        model_name (str): name of model
        stream (str): stream to look at
        args (torch.Tensor): shape (samples, layer, num_neurons)

    Returns:
        list: shape (samples, layer, num_neurons) descriptions
    """
    args_desc = []
    for sample_idx in range(args.shape[0]):
        sample_desc = []
        for layer_idx in range(args.shape[1]):
            layer_desc = []
            for neuron_idx in range(args.shape[2]):
                print(f"sample {sample_idx}, layer {layer_idx}, neuron {neuron_idx}")
                neuron = args[sample_idx, layer_idx, neuron_idx]
                neuron_desc = fetch_neuron_description(model_name, layer_idx, neuron, stream)
                layer_desc.append(neuron_desc)
            sample_desc.append(layer_desc)
        args_desc.append(sample_desc)
    return args_desc

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
    