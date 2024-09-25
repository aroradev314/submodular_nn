import torch
from torch import nn
from icnn import ICNN


# Implements a non-monotone submodular net
# The network is modeled through a concave function composed wiht a modular function
# The modular function is currently being modeled with one layer, but it could be modeled with multiple?
class SubmodularNet(nn.Module):
    def __init__(self, icnn_layers, m_layers, input_size):
        super(SubmodularNet, self).__init__()

        self.input_size = input_size
        self.icnn = ICNN("relu", icnn_layers)
        # self.m = nn.Linear(self.input_size, 1)

        # make modular network sequential instead of one linear layer
        weight_dims = list(zip(m_layers[1:], m_layers))
        self.m = nn.Sequential()

        for odim, idim in weight_dims[:-1]:
            self.m.append(nn.Linear(idim, odim))
            self.m.append(nn.LeakyReLU())
            self.m.append(nn.BatchNorm1d(odim))

        last_odim, last_idim = weight_dims[-1]
        self.m.append(nn.Linear(last_idim, last_odim))
        # self.m.append(nn.Softplus())

    # concave function applied to a positive modular function
    def forward(self, x):
        for layer in self.m:
            x = layer(x)
        x = torch.exp(x)  # to ensure that the output is positive

        return -self.icnn(x)

    def clamp_weights(self):
        with torch.no_grad():
            self.icnn.clamp_weights()
