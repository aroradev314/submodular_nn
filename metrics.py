import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import random
from torch.utils.data import Dataset, DataLoader
from dqn import IncreasingConcaveNet, concavity_regularizer
import math
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
        # d: feature dimension of each element in V
        self.d = V.shape[1]
        # N: number of elements in the global ground set V
        self.N = V.shape[0]
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
            # x is now a binary indicator over the global set V (length N)
            # with 1.0 at selected indices and 0.0 elsewhere
            x = torch.zeros(self.N, dtype=torch.float)
            if len(indices) > 0:
                x[torch.tensor(indices, dtype=torch.long)] = 1.0
            y = self.f(S) if function_name != "fl" and function_name != "gcut" else self.f(S, V)
            self.data.append(x)
            self.labels.append(y)

        self.data = torch.stack(self.data)
        self.labels = torch.tensor(self.labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MonotoneSubmodularSetNet(nn.Module):
    # repr_size is the length of the vector that has the set representation
    def __init__(self, phi_layers, lamb, m_layers, m_size=1, device="cpu"):
        super(MonotoneSubmodularSetNet, self).__init__()
        self.lamb = lamb
        self.m_layers = m_layers

        self.m = nn.ModuleList()
        self.phi = nn.ModuleList()
        # self.phi = IncreasingConcaveNet(phi_layers, device=device)
        # Try: One phi layer for every m layer
        for i in range(m_layers):
            self.phi.append(IncreasingConcaveNet(phi_layers, device=device))
            layers = []
            layers.append(nn.Linear(m_size, 1))
            layers.append(nn.ReLU())
            self.m.append(nn.Sequential(*layers))
        
    def forward(self, x):
        batch_x = x
        ret = torch.sum(self.m[0](batch_x), dim=1)
        # print(f"0: {ret}")
        
        for i in range(1, self.m_layers):
            ret = self.phi[i]((self.lamb * torch.sum(self.m[i](batch_x), dim=1) + (1 - self.lamb) * ret).unsqueeze(-1))

        return ret

    def clamp_weights(self):
        with torch.no_grad():
            # for net in self.phi:
            #     net.clamp_weights()
            self.phi.clamp_weights()


def train(function_name="log", learning_rate=1e-3):
    n = int(1e4)
    d = 10
    V = torch.rand(n, d) # we have n elements, each of which are from 0 to 1 in d dimensions
    # print(V[0])

    dataset = SubmodularSetDataset(V, function_name, subset_sizes=(5, 25), n_subsets=2000)
    batch_size = 32

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    batch_X, batch_y = next(iter(train_loader))
    x0, y0 = batch_X[1], batch_y[1]
    print(x0.shape, y0.shape)
    assert(False)

    example_subset, example_value = dataset[0]
    # print(example_subset)
    # print(example_value)
    # assert(False)

    phi_layers = [1, 16, 16, 1]  # example for IncreasingConcaveNet
    lamb = 0.5
    m_layers = 2
    # when using binary global-set indicators, input size is N (|V|)
    m_size = dataset.N

    model = MonotoneSubmodularSetNet(phi_layers, lamb, m_layers, m_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_reg_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            if (epoch % 10 == 0):
                print(f"Outputs: {outputs[:5].squeeze()}")
                print(f"Batch y: {batch_y[:5]}")

            loss = criterion(outputs.squeeze(), batch_y)
            # reg_loss = (concavity_regularizer(model.phi, strength=epoch, func="square") +
            #             concavity_regularizer(model.m, strength=epoch, func="square"))
            reg_loss = concavity_regularizer(model.phi, strength=epoch, func="square")
            # print(loss)
            reg_loss = 0
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()


            running_loss += total_loss.item() * batch_X.size(0)
            total_reg_loss += reg_loss * batch_X.size(0)
            # if epoch >= first_hard_enforced:
            #     model.clamp_weights()
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_reg_loss = total_reg_loss / len(train_dataset)

        # Evaluate on test set

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                # print(batch_X, outputs)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        test_loss = test_loss / len(test_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        # wandb.log({"epoch": epoch + 1, "Train Loss": epoch_loss, "Test Loss": test_loss})
        # time.sleep(2)

if __name__ == "__main__":
    train()




