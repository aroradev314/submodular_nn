import torch
from torch import nn
from increasing_concave_net import IncreasingConcaveNet


class SubmodularMonotoneNet(nn.Module):
    def __init__(self, layers, lamb, input_size, sample_len, phi_layers, dataset):
        super(SubmodularMonotoneNet, self).__init__()
        self.layers = layers
        self.lamb = lamb

        self.input_size = input_size
        self.sample_len = sample_len
        self.phi_layers = phi_layers

        self.phi = IncreasingConcaveNet(self.phi_layers)

        self.m = nn.ModuleList()

        for i in range(self.layers):
            self.m.append(nn.Linear(self.sample_len, 1))

        self.dataset = dataset

    def forward(self, x):
        precomputed = torch.zeros(self.layers, self.input_size, dtype=torch.float)

        for i in range(self.layers):
            precomputed[i] = (self.m[i](self.dataset.samples)).squeeze()

        cur = torch.sum(precomputed[0] * x, dim=1)

        for i in range(1, self.layers):
            before = (
                self.lamb * cur + (1 - self.lamb) * torch.sum(precomputed[i] * x, dim=1)
            ).T
            cur = self.phi(before).squeeze()

        return cur

    def clamp_weights(self):
        with torch.no_grad():
            self.phi.clamp_weights()

            for i in range(self.layers):
                self.m[i].weight.clamp_(0, torch.inf)
