import numpy as np
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import h5py as hp
from utils import load_model, embed, pca
import pandas as pd

MODEL_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
N_LAYER = 24
N_EMBD = 1024
raw_data_name = "../data/dataset.csv"
dataset_name = "../data/dataset.h5"

print(f"\nLoading {MODEL_FILE}...")

weights = load_model(MODEL_FILE)
tokenizer = Tokenizer.from_file("../model/20B_tokenizer.json")

print("Loading texts...")

data = pd.read_csv(raw_data_name)

print(f"\nEmbedding text...")

embs = np.zeros((len(data), N_EMBD), np.float32)

for i, text in enumerate(data["Title"]):
    embs[i] = embed(text, weights, tokenizer, N_LAYER, N_EMBD)
    print(f"\n{i + 1}/{len(data)} - \"{text}\":\n{np.around(embs[i], 3)}\n")

print("Saving embeddings...")

with hp.File(dataset_name) as file:
    file.create_dataset("headlines", data=embs)
    file.create_dataset("labels", data=data["Value"])

print("Creating visualization...")

pca_result = pca(embs, 3)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])

for i, text in enumerate(data["Title"]):
  ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], text)

ax.set_title("PCA visualization of embeddings")

print("Done.")

plt.show()
