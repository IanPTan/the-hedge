import numpy as np
import torch as pt
from model import RWKV
from tokenizers import Tokenizer

def dumb_simi(a, b):

  return ((a - b) ** 2).sum()


def cos_simi(a, b):

  dot_prod = a @ b
  a_mag = (a ** 2).sum()
  b_mag = (b ** 2).sum()

  return dot_prod / (a_mag * b_mag)


def load_model(MODEL_FILE, device="cpu"):

    state_dict = pt.load(MODEL_FILE, map_location=device)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].squeeze()
        state_dict[k] = state_dict[k].float()

    return state_dict


class Embedder():

    def __init__(self, TOKENIZER_FILE, MODEL_FILE, N_LAYER=24, device="cpu"):

        self.device = device
        state_dict = load_model(MODEL_FILE=MODEL_FILE, device=device)
        self.tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
        self.rwkv = RWKV(state_dict, N_LAYER)
        self.rwkv.eval()

    def __call__(self, text):

        max_length = 0
        encodings = self.tokenizer.encode_batch(text)
        for i, encoding in enumerate(encodings):
            if len(encoding) > max_length:
                max_length = len(encoding)
            encodings[i] = pt.tensor(encoding.ids, device=self.device)

        x = pt.zeros((len(text), max_length), device=self.device, dtype=pt.int)
        lengths = []
        for i, encoding in enumerate(encodings):
            length = len(encoding)
            x[i, :length] = encoding
            lengths.append(length)
        lengths = pt.tensor(lengths, dtype=pt.int) - 1

        self.rwkv.reset()
        y = self.rwkv(x, lengths)

        return y


def pca(X, num_components):

  X_meaned = X - np.mean(X, axis=0)
  
  cov_mat = np.cov(X_meaned, rowvar=False)
  
  eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
  
  sorted_index = np.argsort(eigenvalues)[::-1]
  sorted_eigenvectors = eigenvectors[:, sorted_index]
  
  eigenvector_subset = sorted_eigenvectors[:, :num_components]
  
  X_reduced = np.dot(X_meaned, eigenvector_subset)

  return X_reduced


if __name__ == "__main__":
    e = Embedder("../model/20B_tokenizer.json", "../model/RWKV-4-Pile-430M-20220808-8066.pth")
    y = e(["test", "bruh what"])
