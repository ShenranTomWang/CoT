import torch
import os
from utils.data_collection_utils import get_layer_acts_post_resid, get_layer_acts_attn, get_layer_acts
from utils.loading_utils import load_model
torch.set_grad_enabled(False)

INPUT = os.getenv("INPUT")
MODEL = os.getenv("MODEL")
STREAM = os.getenv("STREAM")
OUTPUT = os.getenv("OUTPUT", f"./experimental_data/{MODEL}/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT, exist_ok=True)

if __name__ == "__main__":
    model = load_model(MODEL, device, dtype=torch.bfloat16)
    print(model)
    layers = range(len(model.blocks))
    
    if STREAM == "res":
        _, acts_resid = get_layer_acts_post_resid([INPUT], model, layers)
        acts_resid = torch.stack([acts_resid[key] for key in acts_resid.keys()], dim=0)
        acts_resid = acts_resid[:, 0, :]
        torch.save(acts_resid, OUTPUT + "acts_res.pt")
    elif STREAM == "attn":
        _, acts_q, acts_k, acts_v = get_layer_acts_attn([INPUT], model, layers)
        acts_q = torch.stack([acts_q[key] for key in acts_q.keys()], dim=0)
        acts_k = torch.stack([acts_k[key] for key in acts_k.keys()], dim=0)
        acts_v = torch.stack([acts_v[key] for key in acts_v.keys()], dim=0)
        acts_q = acts_q[:, 0, :, :]
        acts_k = acts_k[:, 0, :, :]
        acts_v = acts_v[:, 0, :, :]
        torch.save(acts_q, OUTPUT + "acts_q.pt")
        torch.save(acts_k, OUTPUT + "acts_k.pt")
        torch.save(acts_v, OUTPUT + "acts_v.pt")
    else:
        _, acts = get_layer_acts([INPUT], model, layers, STREAM)
        acts = torch.stack([acts[key] for key in acts.keys()], dim=0)
        torch.save(acts, OUTPUT + f"acts_{STREAM}.pt")
        