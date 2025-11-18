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
import os

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

def graph_cut_fn(S, V, sigma=0.1):
    # F(S) = sum_{u in V, v in S} u^T v - sigma * sum_{u,v in S} u^T v
    # V_mat = torch.stack(V)  # (|V|, d)
    print(len(S))
    S_mat = torch.stack(S)  # (|S|, d)
    
    cut = (V @ S_mat.T).sum()
    internal = (S_mat @ S_mat.T).sum()

    return cut - sigma * internal
    # cut = sum(torch.dot(u, v) for u in V for v in S)
    # ternal = sum(torch.dot(u, v) for u in S for v in S)

def monotone_graph_cut_fn(S, V):
    return graph_cut_fn(S, V, sigma=0.1)

# graph cut with higher sigma for non-monotone behavior
def non_monotone_graph_cut_fn(S, V):
    return monotone_graph_cut_fn(S, V, sigma=0.8)

class SubmodularSetDataset(Dataset):
    """Dataset of submodular subsets using prefix generation.

    This dataset always generates exactly N examples by taking a random
    permutation of V and using each prefix as a subset: (V_1), (V_1,V_2), ...
    (V_1..V_N).

    Args:
        V (torch.Tensor): (N, d) tensor of element features.
        function_name (str): which function to evaluate ('log','logdet','fl','gcut').
        precomputed_data (torch.Tensor or None): optional precomputed x matrix (num_examples, N).
        precomputed_labels (torch.Tensor or None): optional precomputed labels (num_examples,).
        use_binary (bool): whether to use binary global set indicators (default True).
        seed (int|None): optional random seed to make the permutation reproducible.
    """
    def __init__(self, V, function_name, precomputed_data=None, precomputed_labels=None, use_binary=True, seed=None):
        self.V = V
        # d: feature dimension of each element in V
        self.d = V.shape[1]
        # N: number of elements in the global ground set V
        self.N = V.shape[0]
        self.use_binary = use_binary

        self.function_name = function_name
        funcs = {
            "log": log_fn,
            "logdet": logdet_fn,
            "fl": facility_location_fn,
            "monotone_gcut": monotone_graph_cut_fn,
            "non_monotone_gcut": non_monotone_graph_cut_fn
        }
        self.f = funcs[function_name]

        # If precomputed data provided, use it directly
        if precomputed_data is not None and precomputed_labels is not None:
            self.data = precomputed_data
            self.labels = precomputed_labels
            self.n_subsets = len(self.data)
            return

        # optionally set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # generate exactly N examples using a random permutation of V and prefixes
        data_list = []
        labels_list = []
        perm = torch.randperm(self.N).tolist()
        for k in range(1, self.N + 1):
            indices = perm[:k]
            S = [self.V[i] for i in indices]
            if self.use_binary:
                x = torch.zeros(self.N, dtype=torch.float)
                if len(indices) > 0:
                    x[torch.tensor(indices, dtype=torch.long)] = 1.0
            else:
                x = torch.stack(S).sum(dim=0) if len(S) > 0 else torch.zeros(self.d)
            y = self.f(S) if function_name != "fl" and ("gcut" not in function_name) else self.f(S, self.V)
            data_list.append(x)
            labels_list.append(y)

        # stack into tensors and set n_subsets to N for compatibility
        self.data = torch.stack(data_list)
        self.labels = torch.tensor(labels_list).float()
        self.n_subsets = self.N

    def save(self, path):
        """Save dataset to disk (creates parent dir if needed)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'V': self.V,
            'function_name': self.function_name,
            'use_binary': self.use_binary,
            'data': self.data,
            'labels': self.labels,
        }, path)

    @classmethod
    def load_from_file(cls, path):
        """Load a saved dataset from disk and return a SubmodularSetDataset instance."""
        d = torch.load(path)
        ds = cls(d['V'], d['function_name'], precomputed_data=d.get('data'),
                 precomputed_labels=d.get('labels'), use_binary=d.get('use_binary', True))
        return ds

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
            self.m.append(nn.Linear(m_size, 1, bias=False))
        
    def forward(self, x):
        batch_x = x
        ret = torch.sum(self.m[0](batch_x), dim=1)
        # print(f"0: {ret}")
        
        # print(0, ret.shape)
        for i in range(1, self.m_layers):
            ret = self.phi[i]((self.lamb * torch.sum(self.m[i](batch_x), dim=1) + (1 - self.lamb) * ret).unsqueeze(-1))
            ret = ret.squeeze()
            # print(i, ret.shape)

        return ret

    def clamp_weights(self, hard_enforce=False):
        with torch.no_grad():
            if hard_enforce:
                for net in self.phi:
                    net.clamp_weights()
            # self.phi.clamp_weights()
            for layer in self.m:
                layer.weight.data.clamp_(0)

def train(function_name, learning_rate=1e-3, dataset_path=None, regenerate_dataset=False):
    n = int(1e4)
    d = 10
    V = torch.rand(n, d) # we have n elements, each of which are from 0 to 1 in d dimensions
    # print(V[0])

    # dataset caching: load if path exists and not regenerating
    if dataset_path is not None and os.path.exists(dataset_path) and not regenerate_dataset:
        print(f"Loading dataset from {dataset_path}")
        dataset = SubmodularSetDataset.load_from_file(dataset_path)
    else:
        dataset = SubmodularSetDataset(V, function_name)
        if dataset_path is not None:
            print(f"Saving generated dataset to {dataset_path}")
            dataset.save(dataset_path)
    batch_size = 32

    # split into train / val / test = 1/3 each
    total_len = len(dataset)
    base = total_len // 3
    lengths = [base, base, total_len - 2 * base]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    phi_layers = [1, 100, 100, 1]  # example for IncreasingConcaveNet
    lamb = 0.5
    m_layers = 2
    # input size depends on representation
    m_size = dataset.N if getattr(dataset, 'use_binary', True) else dataset.d

    model = MonotoneSubmodularSetNet(phi_layers, lamb, m_layers, m_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    wandb.init(project="submodular_nn", config={"function": function_name, "phi_layers": phi_layers, "lamb": lamb})

    source_code = wandb.Artifact("source_code", type="python-file")
    source_code.add_file(__file__)
    source_code.add_file("dqn.py")
    wandb.log_artifact(source_code)

    dataset = wandb.Artifact(f"{function_name}_dataset", type="dataset")
    dataset.add_file(dataset_path)
    wandb.log_artifact(dataset)

    num_epochs = int(3e2)

    # first_hard_enforced = num_epochs // 2
    first_hard_enforced = 0
    cfg = {
        "function": function_name,
        "phi_layers": phi_layers,
        "lamb": lamb,
        "m_layers": m_layers,
        "m_size": m_size,
        "use_binary": getattr(dataset, 'use_binary', True),
        "n": n,
        "d": d,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    "num_epochs": num_epochs,
        "dataset_path": dataset_path,
    }
    # update the run config (allows changing values later)
    wandb.config.update(cfg, allow_val_change=True)

    # watch the model to log gradients and parameter histograms
    try:
        wandb.watch(model, log="all", log_freq=100)
    except Exception:
        # guard in case wandb isn't configured to watch this model
        pass

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        total_reg_loss = 0.0
        # accumulate plain MSE (sum over examples) separately so RMSE ignores regularizer
        running_mse_sum = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            # if (epoch % 10 == 0):
            #     print(f"Outputs: {outputs[:5].squeeze()}")
            #     print(f"Batch y: {batch_y[:5]}")

            loss = criterion(outputs.squeeze(), batch_y)
            # reg_loss = (concavity_regularizer(model.phi, strength=epoch, func="square") +
            #             concavity_regularizer(model.m, strength=epoch, func="square"))
            # reg_loss = concavity_regularizer(model.phi, strength=epoch, func="square")
            # print(loss)
            reg_loss = 0
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            # accumulate totals: total_loss for objective, but plain MSE separately
            running_loss += total_loss.item() * batch_X.size(0)
            running_mse_sum += loss.item() * batch_X.size(0)
            total_reg_loss += reg_loss * batch_X.size(0)

            model.clamp_weights(hard_enforce=(epoch >= first_hard_enforced))

        epoch_loss = running_loss / len(train_dataset)
        epoch_reg_loss = total_reg_loss / len(train_dataset)
        # compute train RMSE (disregarding concavity regularizer)
        epoch_mse = running_mse_sum / len(train_dataset)
        epoch_rmse = math.sqrt(epoch_mse)

        # Evaluate on validation set (used for monitoring during training)
        model.eval()
        val_total_loss = 0.0
        val_mse_sum = 0.0
        val_reg_sum = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                mse_loss = criterion(outputs.squeeze(), batch_y)
                reg_loss = concavity_regularizer(model.phi, strength=epoch, func="square")
                val_total_loss += (mse_loss.item() + reg_loss) * batch_X.size(0)
                val_mse_sum += mse_loss.item() * batch_X.size(0)
                val_reg_sum += reg_loss * batch_X.size(0)
        val_loss = val_total_loss / len(val_dataset)
        val_mse = val_mse_sum / len(val_dataset)
        val_rmse = math.sqrt(val_mse)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f} | Train RMSE: {epoch_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "Train Loss": epoch_loss,
            "Val Loss": val_loss,
            "Train RMSE": epoch_rmse,
            "Val RMSE": val_rmse,
        })
        # time.sleep(2)
    
    for i in model.m:
        print(i.weight)
        print(torch.all(i.weight >= 0).item())

    # Final evaluation on the test set: compute both regular loss (MSE + reg)
    # and RMSE (MSE only)
    model.eval()
    test_total_loss = 0.0
    test_mse_sum = 0.0
    test_reg_sum = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            mse_loss = criterion(outputs.squeeze(), batch_y)
            reg_loss = concavity_regularizer(model.phi, strength=num_epochs, func="square")
            test_total_loss += (mse_loss.item() + reg_loss) * batch_X.size(0)
            test_mse_sum += mse_loss.item() * batch_X.size(0)
            test_reg_sum += reg_loss * batch_X.size(0)
            # print the expected vs predicted for first 5 in batch
            print("Predicted vs Actual:")
            for pred, actual in zip(outputs.squeeze()[:5], batch_y[:5]):
                print(f"{pred.item():.4f} vs {actual.item():.4f}")
    test_loss_final = test_total_loss / len(test_dataset)
    test_mse_final = test_mse_sum / len(test_dataset)
    test_rmse_final = math.sqrt(test_mse_final)

    print(f"Final Test - Regular Loss (MSE+reg): {test_loss_final:.4f}, Test RMSE: {test_rmse_final:.4f}")
    wandb.log({
        "Final Test Loss": test_loss_final,
        "Final Test RMSE": test_rmse_final,
    })

if __name__ == "__main__":
    function_name = "monotone_gcut"  # choose from 'log', 'logdet', 'fl', 'gcut'
    dataset_path = os.path.join(os.path.dirname(__file__), "cached_datasets", f"{function_name}.pt")
    train(dataset_path=dataset_path, function_name=function_name, learning_rate=1e-3, regenerate_dataset=True)




