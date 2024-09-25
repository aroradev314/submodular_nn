from torch import nn
import torch


class ICNN(nn.Module):
    def __init__(self, activ, layers, scale=None, device="cpu"):
        super(ICNN, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.As = nn.ParameterList()
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        self.scale = scale
        first_idim = weight_dims[0][1]
        self.layers = layers
        self.device = device
        for odim, idim in weight_dims:
            self.As.append(
                nn.Parameter(
                    torch.tensor(
                        torch.normal(0.0, 1.0, size=(odim, first_idim)),
                        device=self.device,
                    )
                )
            )
            self.Ws.append(
                nn.Parameter(
                    torch.tensor(torch.rand(size=(odim, idim)), device=self.device)
                )
            )
            self.bs.append(
                nn.Parameter(
                    torch.tensor(
                        torch.normal(0.0, 1.0, size=(odim,)), device=self.device
                    )
                )
            )
        if activ == "relu":
            self.activ = nn.ReLU()
        elif activ == "linear":
            self.activ = lambda x: x
        else:
            raise Exception("Unsupported activation function")

    def forward(self, z):
        z0 = z.clone()
        layers = list(zip(self.As, self.Ws, self.bs))

        for A, W, b in layers[:-1]:
            z = self.activ(z0 @ torch.t(A) + z @ torch.t(W) + b)

        out_A, out_W, out_b = layers[-1]
        z = z0 @ torch.t(out_A) + z @ torch.t(out_W) + out_b
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
