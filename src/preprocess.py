import numpy as np
import torch as pt
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import h5py as hp
from utils import Embedder, pca
import pandas as pd
from tqdm import tqdm

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
N_LAYER = 24
N_EMBD = 1024
raw_data_name = "../data/dataset.csv"
dataset_name = "../data/dataset.h5"
batch_size = 128

print(f"Loading {TOKENIZER_FILE} and {RWKV_FILE}...")
embed = Embedder(TOKENIZER_FILE, RWKV_FILE, N_LAYER, device=device)

print("Loading texts...")
data = pd.read_csv(raw_data_name)
data_len = len(data)


print(f"Embedding {data_len} texts...")
embs = np.zeros((data_len, N_EMBD), np.float32)
batch_amnt = data_len // batch_size + (data_len % batch_size > 0)
for start in tqdm(range(0, data_len, batch_size), desc="Embedding...", unit="batch"):
    batch = slice(start, start + batch_size)
    text = data["Title"][batch]
    embs[batch] = embed(text).cpu()

print("Saving embeddings...")
with hp.File(dataset_name, "w") as file:
    file.create_dataset("headlines", data=embs)
    file.create_dataset("labels", data=data["Value"])

"""
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
"""
