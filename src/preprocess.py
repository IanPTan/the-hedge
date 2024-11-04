import numpy as np
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import h5py as hp
from utils import Embedder, pca
import pandas as pd
from tqdm import tqdm

MODEL_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
N_LAYER = 24
N_EMBD = 1024
raw_data_name = "../data/dataset.csv"
dataset_name = "../data/dataset.h5"
batch_size = 128

print(f"Loading {TOKENIZER_FILE} and {MODEL_FILE}...")

embed = Embedder(TOKENIZER_FILE, MODEL_FILE, N_LAYER)

print("Loading texts...")

data = pd.read_csv(raw_data_name)


embs = np.zeros((len(data), N_EMBD), np.float32)

batch_amnt = len(data) // batch_size + (len(data) % batch_size > 0)
for i in tqdm(range(batch_amnt), desc="Embedding...", unit="batch"):
    text = data["Title"][i: i + batch_size]
    embs[i: i + batch_size] = embed(text)

print("Saving embeddings...")

with hp.File(dataset_name, "w") as file:
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
