import torch, os

BATCH = int(os.getenv("BATCH"))
MODEL_NAME = os.getenv("MODEL_NAME")
DATASET = os.getenv("DATASET")
DATA_DIR = f"./experimental_data/{MODEL_NAME}/{DATASET}/"
torch.set_grad_enabled(False)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

diffs_resid = []
acts_resid = []
acts_exp_resid = []
for batch in range(BATCH):
    diffs_resid.append(torch.load(DATA_DIR + f"diffs_resid_{batch}.pt", map_location=device))
    acts_resid.append(torch.load(DATA_DIR + f"acts_resid_{batch}.pt", map_location=device))
    acts_exp_resid.append(torch.load(DATA_DIR + f"acts_exp_resid_{batch}.pt", map_location=device))

diffs_resid = torch.cat(diffs_resid, dim=0)
acts_resid = torch.cat(acts_resid, dim=0)
acts_exp_resid = torch.cat(acts_exp_resid, dim=0)