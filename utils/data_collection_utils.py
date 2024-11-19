from utils.loading_utils import get_pretrained_model
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from transformers import AutoTokenizer
import torch
import json
import pandas as pd
from tqdm import tqdm
import configparser

CONFIG_PATH = "./config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

def load_data(n_shots: int, data_path: str, data_key: str = "examples", lookup_key: str = "pair_id_lookup") -> tuple:
    with open(data_path, "r") as file:
        dataset = json.load(file)

    examples = dataset[data_key]
    pair_id_lookup = dataset[lookup_key] if lookup_key != None else None
    dataset = pd.DataFrame(examples)
    train = dataset.sample(n_shots)
    test = dataset.drop(train.index)

    return train, test, pair_id_lookup

def load_model(model_name: str, device='cpu') -> HookedTransformer:
    print(f"Loading model {model_name}...")
    weights_directory = "./models/" + config[model_name]['weights_directory']
    model = get_pretrained_model(weights_directory, dtype=torch.bfloat16, device=device, verbose=True)
    return model

def load_tokenizer(model_name: str, device='cpu') -> AutoTokenizer:
    print(f"Loading tokenizer {model_name}...")
    weights_directory = "./models/" + config[model_name]['weights_directory']
    model = AutoTokenizer.from_pretrained(weights_directory, dtype=torch.bfloat16, device=device)
    return model

def get_layer_acts_post_resid(statements, model: HookedTransformer, layers: list) -> dict:
    """
    Get given layer post residual activations for the statements. Activations are obtained after the last token is read.
    args:
        statements: The statements to obtain activations for.
        model: The model to use.
        layers: The layers (int) to obtain activations for as a list.
    returns: dictionary of stacked activations of shape (batch_size, hidden_channels)
    """
    acts = {}
    def get_act(value: torch.Tensor, hook: HookPoint):
        acts[hook.name] = value[:, -1, :]

    hooks = []
    for layer in layers:
        hooks.append((get_act_name("resid_post", layer=layer), get_act))

    out = model.run_with_hooks(statements, fwd_hooks=hooks, return_type="logits")

    return out, acts

def get_layer_acts_attn(statements, model: HookedTransformer, layers: list) -> tuple:
    """
    Get given layer attention activations for the statements. Activations are obtained after the last token is read.
    args:
        statements: The statements to obtain activations for.
        model: The model to use.
        layers: The layers (int) to obtain activations for as a list.
    returns: tuple of dictionary of stacked q, k, v activations of shape (batch_size, n_attn_heads, d_k(headdim))
    """
    acts_q = {}
    acts_k = {}
    acts_v = {}
    def get_act_q(value: torch.Tensor, hook: HookPoint):
        acts_q[hook.name] = value[:, -1, :, :]
    def get_act_k(value: torch.Tensor, hook: HookPoint):
        acts_k[hook.name] = value[:, -1, :, :]
    def get_act_v(value: torch.Tensor, hook: HookPoint):
        acts_v[hook.name] = value[:, -1, :, :]

    hooks = []
    for layer in layers:
        hooks.append((get_act_name("q", layer=layer), get_act_q))
        hooks.append((get_act_name("k", layer=layer), get_act_k))
        hooks.append((get_act_name("v", layer=layer), get_act_v))

    _ = model.run_with_hooks(statements, fwd_hooks=hooks, return_type=None)

    return acts_q, acts_k, acts_v

def top_p_sampling(logits, p=0.9):
    """
    Implements top-p (nucleus) sampling.
    
    Args:
        logits: The logits output from the model (shape: [vocab_size]).
        p: The cumulative probability threshold for top-p sampling.
        
    Returns:
        The index of the sampled token.
    """
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Sort probabilities and their indices in descending order
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens with cumulative probabilities above p
    cutoff_index = torch.where(cumulative_probs > p)[0][0]  # First index where cumulative prob > p
    sorted_probs = sorted_probs[:cutoff_index + 1]
    sorted_indices = sorted_indices[:cutoff_index + 1]

    # Normalize probabilities to sum to 1
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample from the filtered distribution
    sampled_index = torch.multinomial(sorted_probs, num_samples=1).item()

    # Return the original index of the sampled token
    return sorted_indices[sampled_index]

def generate_top_p_with_hooks(query: str, model: HookedTransformer, layers: list, tokenizer: AutoTokenizer, p: float = 0.9) -> tuple:
    """top p generation pass that obtains activations

    Args:
        query (str): query input
        model (HookedTransformer)
        layers (list)
        tokenizer (AutoTokenizer)
        p (float, optional): for top-p sampling. Defaults to 0.9.

    Returns:
        tuple: generations, acts_resid
    """
    output = ""
    acts_resid = []
    
    while output != "<eos>":
        logits, act_resid = get_layer_acts_post_resid(query, model, layers)
        logit = logits[:, -1, :]
        next_id = top_p_sampling(logit, p=p)
        next_token = tokenizer.decode(next_id)
        output = output + " " + next_token
        print(output)
        acts_resid.append(act_resid)
    
    return query, acts_resid

def generate_greedy_with_hooks(query: str, model: HookedTransformer, layers: list, tokenizer: AutoTokenizer) -> tuple:
    """greedy generation pass that obtains activations

    Args:
        query (str): query input
        model (HookedTransformer)
        layers (list)
        tokenizer (AutoTokenizer)

    Returns:
        tuple: generations, acts_resid
    """
    output = ""
    acts_resid = []
    
    while output != "<eos>":
        logits, act_resid = get_layer_acts_post_resid(query, model, layers)
        logit = logits[0, -1, :]
        # print(logit.shape)
        next_id = torch.argmax(logit, dim=-1)
        output = tokenizer.decode(next_id)
        query = query + output
        acts_resid.append(act_resid)
        print(query)
    
    return query, acts_resid

def obtain_single_line_generation_act(
    model: HookedTransformer,
    query: str,
    exp: str,
    layers: list,
    train_prompt: str,
    tokenizer: AutoTokenizer,
    p: float = 0.9
) -> tuple:
    """obtain activation difference between query and query + exp.

    Args:
        model (HookedTransformer): model
        query (str)
        exp (str): experimental string
        layers (list): layers to obtain activations
        train_prompt (str): prompt added before query
        p (int): p for top p sampling, defaults to 0.9

    Returns:
        tuple: tuple<list<torch.Tensor>> the activation of query and query + exp at each timestep, as well as generated output at each timestep
    """
    query = train_prompt + query
    query = query
    query_exp = query + exp
    query_exp = query_exp
    
    generations, acts_resid = generate_greedy_with_hooks(query, model, layers, tokenizer)
    generations_exp, acts_exp_resid = generate_greedy_with_hooks(query_exp, model, layers, tokenizer)
        
    return acts_resid, acts_exp_resid, generations, generations_exp


def obtain_act_diff(
    model: HookedTransformer, 
    queries: pd.DataFrame, 
    batch_size: int, 
    exp: str, 
    layers: list, 
    train_prompt: str, 
    prompt_key: str = "sent", 
    start_idx: int = 0,
    max_idx: int = -1
) -> tuple:
    """
    Obtains the activation difference between queries and queries + exp.
    args:
        model: The model to use.
        queries: The queries to use.
        batch_size: The batch size to use.
        exp: The prompt to experiment with, added to the end of the sentence.
        layers: The list of layers (int) to obtain diff.
        train_prompt: The prompt used to train the model.
        prompt_key: The key in the queries dataframe that contains the prompt.
        start_idx: The starting index of data collection. Set to 0 by default.
        max_idx: The maximum number of queries to use. Set to -1 to use all queries.
    returns:
        The activation difference between the model's predictions for the given queries, original activations, experimental activations.
        tuple<list<map<str, torch.Tensor>>>
    """
    diffs_resid = []
    acts_resid = []
    acts_resid_exp = []
    diffs_q, diffs_k, diffs_v = [], [], []
    acts_q, acts_k, acts_v = [], [], []
    acts_q_exp, acts_k_exp, acts_v_exp = [], [], []
    max_idx = len(queries) if max_idx == -1 else max_idx
    for batch_idx in tqdm(range(start_idx, max_idx, batch_size), desc="Processing batches"):
        batch = queries.iloc[batch_idx : batch_idx + batch_size][prompt_key].tolist()
        batch = [train_prompt + query for query in batch]
        batch_exp = [train_prompt + query + exp for query in batch]

        _, act_resid = get_layer_acts_post_resid(batch, model, layers)
        _, act_resid_exp = get_layer_acts_post_resid(batch_exp, model, layers)
        acts_resid.append(act_resid)
        acts_resid_exp.append(act_resid_exp)
        diff_resid = {layer: act_resid_exp[layer] - act_resid[layer] for layer in act_resid.keys()}
        diffs_resid.append(diff_resid)

        # act_q, act_k, act_v = get_layer_acts_attn(batch, model, layers)
        # act_q_exp, act_k_exp, act_v_exp = get_layer_acts_attn(batch_exp, model, layers)
        # diff_q = {layer: act_q_exp[layer] - act_q[layer] for layer in act_q.keys()}
        # diff_k = {layer: act_k_exp[layer] - act_k[layer] for layer in act_k.keys()}
        # diff_v = {layer: act_v_exp[layer] - act_v[layer] for layer in act_v.keys()}
        # diffs_q.append(diff_q)
        # diffs_k.append(diff_k)
        # diffs_v.append(diff_v)
        # acts_q.append(act_q)
        # acts_k.append(act_k)
        # acts_v.append(act_v)
        # acts_q_exp.append(act_q_exp)
        # acts_k_exp.append(act_k_exp)
        # acts_v_exp.append(act_v_exp)

    return diffs_resid, acts_resid, acts_resid_exp # , diffs_q, acts_q, acts_q_exp, diffs_k, acts_k, acts_k_exp, diffs_v, acts_v, acts_v_exp