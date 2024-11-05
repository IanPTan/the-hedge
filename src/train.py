import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from dataset import Dataset
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
dataset_name = "../data/dataset.h5"
epochs = 1000
batch_size = 101
#batch_size = 26000

model = Model(features=[1024, 512, 3]).to(device)
dataset = Dataset(dataset_name)

batch_len = len(dataset) // batch_size + (len(dataset) % batch_size > 0)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128, eta_min=0)

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
    scheduler.step()
    all_losses[epoch] = pt.mean(losses)
    pbar.set_postfix(loss=f"{pt.mean(losses).item():.4f}")
pt.save(model.state_dict(), "backup.ckpt")
plt.plot(all_losses.detach())
plt.show()
