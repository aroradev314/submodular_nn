from torch import nn
import torch


# Implements increasing concave neural network over its input y
class IncreasingConcaveNet(nn.Module):
    def __init__(self, layers, scale=None, device="cpu"):
        super(IncreasingConcaveNet, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        self.scale = scale
        self.layers = layers
        self.device = device
        for odim, idim in weight_dims:
            self.Ws.append(
                nn.Parameter(
                    torch.tensor(
                        torch.normal(0.0, 1e-6, size=(odim, idim)), device=self.device
                    )
                )
            )
            self.bs.append(
                nn.Parameter(
                    torch.tensor(
                        torch.normal(0.0, 1e-6, size=(odim,)), device=self.device
                    )
                )
            )

        self.activ = lambda x: torch.min(torch.zeros_like(x), x)
        # self.activ = lambda x: torch.ones_like(x) - torch.exp(-x)

    def forward(self, z):
        layers = list(zip(self.Ws, self.bs))

        z = torch.unsqueeze(z, -1)

        for W, b in layers[:-1]:
            z = self.activ(z @ torch.t(W) + b)

        out_W, out_b = layers[-1]
        z = z @ torch.t(out_W) + out_b
        if self.scale is not None:
            return z * self.scale
        else:
            return z

    # MSE Loss
    def loss(self, x, y):
        return torch.sum((self.forward(x) - y) ** 2)

    # make parameters of hidden layers positive to preserve convexity
    def clamp_weights(self):
        for layer in self.Ws:
            layer.data = torch.clamp(layer.data, 0, torch.inf)
