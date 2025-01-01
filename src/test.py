import torch as pt
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import h5py as hp
from utils import Embedder, pca
import pandas as pd
from model import Model

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
device = "cpu"
RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
N_LAYER = 24
N_EMBD = 1024
model_file = "model.ckpt"

print(f"Loading {TOKENIZER_FILE} and {RWKV_FILE}...")
embed = Embedder(TOKENIZER_FILE, RWKV_FILE, N_LAYER)

print(f"Loading {model_file}...")
model = Model(features=[1024, 1024, 1024, 512, 128, 32, 5]).to(device)
#model = Model(features=[1024, 512, 32, 5]).to(device)
weights = pt.load(model_file)
model.load_state_dict(weights)
model.eval()

print(f"Evaluation Mode.")

labels = ["Negative spike", "Negative followed by positive spike", "Positive followed by negative spike", "Positive spike", "No meaningful spikes"]
while 1:
    text = input("\nHeadline: ")
    emb = embed([text])
    pred = model(emb)[0]
    print("\nPrediction:")
    for p, l in zip(pred, labels):
        print(f"{l}: {p * 100:.4f}%")
