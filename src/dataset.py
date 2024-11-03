import numpy as np
import torch as pt
import h5py as hp


class Dataset(pt.utils.data.Dataset):

    def __init__(self, file_path):

        self.file = hp.File(file_path, "r")
        self.headlines = self.file["headlines"]
        self.labels = self.file["labels"]

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return self.headlines[idx], self.labels[idx]

    def __del__(self):

        self.file.close()
