import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from dataset import Dataset
from model import Model
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import h5py as hp
from grokfast_pytorch import GrokFastAdamW


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
dataset_name = "dataset.h5"
epochs = 10000
#batch_size = 2 ** 12
batch_size = 22336

model = Model(features=[1024, 1024, 1024, 512, 128, 32, 3]).to(device)
#model = Model(features=[1024, 512, 32, 5]).to(device)
dataset = Dataset(dataset_name)

batch_len = len(dataset) // batch_size + (len(dataset) % batch_size > 0)

class_counts = pt.bincount(pt.tensor(dataset.labels[:])) 
class_weights = 1.0 / class_counts
class_weights = class_weights / pt.sum(class_weights)
weights = [class_weights[label] for label in dataset.labels[:]] 
sampler = WeightedRandomSampler(weights, len(dataset.labels[:]))

#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

criterion = pt.nn.CrossEntropyLoss()
#optimizer = pt.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
optimizer = GrokFastAdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
#scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=0)
scheduler = pt.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)

all_losses = pt.zeros(epochs)
print(f"Starting training using {device}.")
pbar = tqdm(range(epochs), desc="Training...", unit="epoch")
for epoch in pbar:
    losses = pt.zeros(batch_len)
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[i] = loss
        loss.backward()
        optimizer.step()
    scheduler.step(pt.mean(losses))
    all_losses[epoch] = pt.mean(losses)
    pbar.set_postfix(loss=f"{pt.mean(losses).item():.4f}")

pt.save(model.state_dict(), "backup.ckpt")
with hp.File("logs.h5", "w") as file:
    file.create_dataset("loss", data=all_losses.detach())

"""
plt.plot(all_losses.detach())
plt.show()
"""
