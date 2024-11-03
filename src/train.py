import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from dataset import Dataset
from model import Model
from torch.utils.data import DataLoader


device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
dataset_name = "../data/dataset.h5"
epochs = 1000
batch_size = 101
print_freq = 100

model = Model(in_features=1024, out_features=2).to(device)
dataset = Dataset(dataset_name)

batch_len = len(dataset) // batch_size

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = pt.nn.CrossEntropyLoss()
optimizer = pt.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

all_losses = pt.zeros(epochs)
print(f"Starting training using {device}.")
for epoch in range(epochs):
    losses = pt.zeros(batch_len)
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[i] = loss
        loss.backward()
        optimizer.step()
    all_losses[epoch] = pt.mean(losses)
    if (epoch + 1) % print_freq == 0:
        print(f"Epoch {epoch + 1}, Loss: {pt.mean(losses)}")
pt.save(model.state_dict(), "backup.ckpt")
plt.plot(all_losses.detach())
plt.show()
