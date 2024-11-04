import torch as pt
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import h5py as hp
from utils import Embedder, pca
import pandas as pd
from model import Model

RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
N_LAYER = 24
N_EMBD = 1024
model_file = "model.ckpt"

print(f"Loading {TOKENIZER_FILE} and {RWKV_FILE}...")
embed = Embedder(TOKENIZER_FILE, RWKV_FILE, N_LAYER)

print(f"Loading {model_file}...")
model = Model(in_features=1024, out_features=3)
weights = pt.load(model_file)
model.load_state_dict(weights)
model.eval()

print(f"Evaluation Mode.")

while 1:
    text = input("Headline: ")
    emb = embed([text])
    pred = model(emb)[0]
    print(f"Good: {pred[1]}\nNeutral: {pred[2]}\nBad: {pred[0]}")
