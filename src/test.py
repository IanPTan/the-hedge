import torch as pt
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import h5py as hp
from utils import load_model, embed, pca
import pandas as pd
from model import Model

RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
N_LAYER = 24
N_EMBD = 1024
model_file = "model.ckpt"

print(f"\nLoading {RWKV_FILE}...")

rwkv_weights = load_model(RWKV_FILE)
tokenizer = Tokenizer.from_file("../model/20B_tokenizer.json")

model = Model(in_features=1024, out_features=2)
weights = pt.load(model_file)
model.load_state_dict(weights)
model.eval()

print(f"Evaluation Mode.")

while 1:
    text = input("Headline: ")
    emb = embed(text, rwkv_weights, tokenizer, N_LAYER, N_EMBD)
    pred = model(pt.tensor(emb))
    print(f"Good: {pred[1]}\nBad: {pred[0]}")
