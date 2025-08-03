import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from torch.utils.data import Dataset, DataLoader
from dqn import MonotoneSubmodularNet
import wandb

def log_fn(S):
    # F(S) = log(sum_s 1^T z_s)
    total = sum(torch.sum(z) for z in S)
    return torch.log(total + 1e-6)

def logdet_fn(S):
    # F(S) = log det(I + sum_s z_s z_s^T)
    d = S[0].shape[0]
    A = torch.eye(d)
    for z in S:
        A += torch.ger(z, z)  # outer product
    return torch.logdet(A + 1e-6 * torch.eye(d))

def facility_location_fn(S, V):
    # F(S) = sum_{s' in V} max_{s in S} z_s^T z_s' / (||z_s|| ||z_s'||)
    total = 0
    for z0 in V:
        sims = [torch.dot(z, z0) / (torch.norm(z) * torch.norm(z0) + 1e-6) for z in S]
        total += max(sims)
    return total

def monotone_graph_cut_fn(S, V, sigma=0.1):
    # F(S) = sum_{u in V, v in S} u^T v - sigma * sum_{u,v in S} u^T v
    cut = sum(torch.dot(u, v) for u in V for v in S)
    internal = sum(torch.dot(u, v) for u in S for v in S)
    return cut - sigma * internal

class SubmodularSetDataset(Dataset):
    def __init__(self, V, function_name, subset_sizes=(5, 25), n_subsets=2000):
        self.V = V
        self.d = V.shape[1]
        self.data = []
        self.labels = []
        self.function_name = function_name
        funcs = {
            "log": log_fn,
            "logdet": logdet_fn,
            "fl": facility_location_fn,
            "gcut": monotone_graph_cut_fn,
        }
        self.f = funcs[function_name]

        for _ in range(n_subsets):
            k = random.randint(*subset_sizes)
            indices = sorted(random.sample(range(len(V)), k))
            S = [V[i] for i in indices]
            x = torch.stack(S).sum(dim=0)  # sum-pooled representation
            y = self.f(S) if function_name != "fl" and function_name != "gcut" else self.f(S, V)
            self.data.append(x)
            self.labels.append(y)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(model, dataset, lr=1e-3, epochs=50, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            pred = model(x)
            loss = loss_fn(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.clamp_weights()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}")

# Generate V
d = 10
n = int(1e4)
V = [torch.rand(d) for _ in range(n)]

# Choose the function to test
function_name = "log"  # or "logdet", "fl", "gcut"

# Dataset
dataset = SubmodularSetDataset(V, function_name)

# Model
m_layers = 2
phi_layers = [1, 50, 50, 50, 1]
lamb = 0.5
wandb.log({"M layers": m_layers, "Phi layers": phi_layers, "Lambda": lamb})
model = MonotoneSubmodularNet(phi_layers=phi_layers, lamb=lamb, m_layers=m_layers)

# Train
train(model, dataset)

