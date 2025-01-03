import pandas as pd
import torch as pt
import numpy as np
from tokenizers import Tokenizer
from model import Model
import matplotlib.pyplot as plt
import h5py as hp
from utils import Embedder, pca
from tqdm import tqdm


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
model_file = "model.ckpt"
N_LAYER = 24
N_EMBD = 1024
batch_size = 128

print(f"Loading {TOKENIZER_FILE} and {RWKV_FILE}...")
embed = Embedder(TOKENIZER_FILE, RWKV_FILE, N_LAYER, device=device)

print(f"Loading {model_file}...")
model = Model(features=[1024, 1024, 1024, 512, 128, 32, 5]).to(device)
weights = pt.load(model_file)
model.load_state_dict(weights)
model.eval()

results_len = len(results)
print(f"Embedding {results_len} articles...")
embs = pt.zeros((results_len, N_EMBD), dtype=pt.float32, device=device)
batch_amnt = results_len // batch_size + (results_len % batch_size > 0)
for start in tqdm(range(0, results_len, batch_size), desc="Embedding...", unit="batch"):
    batch = slice(start, start + batch_size)
    text = results["title"][batch]
    embs[batch] = embed(text)

print(f"Predicting...") 
preds = model(embs).detach().cpu().numpy()
results[["neg", "neg pos", "pos neg", "pos", "none"]] = preds * 10
results["label"] = preds.argmax(-1)
results.to_csv("scan.all.csv", index=False)

sig_results = results[results["label"] != 4]
sig_results.to_csv("scan.csv", index=False)

print(f"Labeled {len(sig_results)} significant articles from {len(results)} articles.")
