import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from replay_memory import ReplayMemory
from torch import nn
import torch
import math

class IncreasingConcaveNet(nn.Module):
    def __init__(self, layers, device="cpu"):
        super(IncreasingConcaveNet, self).__init__()
        weight_dims = list(zip(layers[1:], layers))
        self.layers = nn.ModuleList()
        self.device = device
        for odim, idim in weight_dims:
            self.layers.append(nn.Linear(idim, odim))
            
        self.activ = lambda x: torch.min(torch.zeros_like(x), x)

    def forward(self, z):
        for layer in self.layers[:-1]:
            z = self.activ(layer(z))

        return self.layers[-1](z)

    # make parameters of hidden layers positive to preserve convexity
    def clamp_weights(self):
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.data.clamp_(min=0)  # clamp only weights, not bias

# an alternative to the traditional approach where we have a soft regularization penalty instead of
# doing a hard weight clamp every time 
def concavity_regularizer(models, strength=1.0, func="linear"):
    penalty = 0.0
    assert(func in ("linear", "square"), "type must either be 'linear' or 'square'")
    power = (1 if func == "linear" else 2)
    for model in models:
        try:
            for layer in model.layers:  # usually only hidden layers must be constrained
                # penalize negative weights
                penalty += torch.pow(torch.sum(torch.relu(-layer.weight)), power) 
        except:
            penalty += torch.pow(torch.sum(torch.relu(-model.weight)), power) 
    return strength * penalty

class MonotoneSubmodularNet(nn.Module):
    # repr_size is the length of the vector that has the set representation
    def __init__(self, phi_layers, lamb, m_layers, m_size=1, device="cpu"):
        super(MonotoneSubmodularNet, self).__init__()
        self.lamb = lamb
        self.m_layers = m_layers

        self.m = nn.ModuleList()
        self.phi = nn.ModuleList()
        # self.phi = IncreasingConcaveNet(phi_layers, device=device)
        # Try: One phi layer for every m layer
        for i in range(m_layers):
            self.phi.append(IncreasingConcaveNet(phi_layers, device=device))
            layers = []
            layers.append(nn.Linear(m_size, 10))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(10, 10))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(10, 1))
            layers.append(nn.Softplus()) # smooth alternative to relu 
            self.m.append(nn.Sequential(*layers))
    
    def forward(self, x):
        batch_x = torch.unsqueeze(x, -1)
        ret = torch.sum(self.m[0](batch_x), dim=1)
        # print(f"0: {ret}")
        
        for i in range(1, self.m_layers):
            ret = self.phi[i](self.lamb * torch.sum(self.m[i](batch_x), dim=1) + (1 - self.lamb) * ret)
            # remove the phi network from the equation
            # ret = torch.log1p(self.lamb * torch.sum(self.m[i](batch_x), dim=1) + (1 - self.lamb) * ret)

            # print(f"{i + 1}: {ret}")

        return ret

    def clamp_weights(self):
        with torch.no_grad():
            # for net in self.phi:
            #     net.clamp_weights()
            self.phi.clamp_weights()
    
class PartialInputConcaveNN(nn.Module):
    def __init__(self, u_layers, z_layers, device="cpu"):
        assert (
            len(z_layers) == len(u_layers) + 1
        ), "there must be 1 more z layer than u layer"
        super(PartialInputConcaveNN, self).__init__()
        self.device = device
        self.act = lambda x : torch.min(torch.zeros_like(x), x)

        self.u_layers_dim = list(
            zip(u_layers[1:], u_layers)
        )  # [[output dim 1, input dim 1], [output dim 2, input dim 2]]
        self.z_layers_dim = list(zip(z_layers[1:], z_layers))
        self.xdim = u_layers[0]
        self.ydim = z_layers[0]

        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        for odim, idim in self.u_layers_dim:
            self.Ws.append(
                nn.Parameter(
                    torch.normal(0.0, 1.0, size=(odim, idim), device=self.device)
                )
            )
            self.bs.append(
                nn.Parameter(torch.normal(0.0, 1.0, size=(odim,), device=self.device))
            )

        self.W_zs = nn.ParameterList()
        self.W_zus = nn.ParameterList()
        self.b_zs = nn.ParameterList()
        self.W_ys = nn.ParameterList()
        self.W_yus = nn.ParameterList()
        self.b_ys = nn.ParameterList()
        self.W_us = nn.ParameterList()
        self.b_us = nn.ParameterList()

        for i, (odim, idim) in enumerate(self.z_layers_dim):
            udim = u_layers[i]
            
            # For W_zs - uniform initialization with k based on input dimension
            W_z = nn.Parameter(torch.zeros(odim, idim, device=self.device))
            k_z = 1.0 / idim
            nn.init.uniform_(W_z, -math.sqrt(k_z), math.sqrt(k_z))
            self.W_zs.append(W_z)
            
            # For W_ys - uniform initialization with k based on input dimension (ydim)
            W_y = nn.Parameter(torch.zeros(odim, self.ydim, device=self.device))
            k_y = 1.0 / self.ydim
            nn.init.uniform_(W_y, -math.sqrt(k_y), math.sqrt(k_y))
            self.W_ys.append(W_y)
            
            # For W_zus - uniform initialization with k based on input dimension (udim)
            W_zu = nn.Parameter(torch.zeros(idim, udim, device=self.device))
            k_zu = 1.0 / udim
            nn.init.uniform_(W_zu, -math.sqrt(k_zu), math.sqrt(k_zu))
            self.W_zus.append(W_zu)
            
            # For b_zs - uniform initialization for bias with k based on input dimension
            b_z = nn.Parameter(torch.zeros(idim, device=self.device))
            nn.init.uniform_(b_z, -math.sqrt(k_z), math.sqrt(k_z))
            self.b_zs.append(b_z)
            
            # For W_yus - uniform initialization with k based on input dimension (udim)
            W_yu = nn.Parameter(torch.zeros(self.ydim, udim, device=self.device))
            k_yu = 1.0 / udim
            nn.init.uniform_(W_yu, -math.sqrt(k_yu), math.sqrt(k_yu))
            self.W_yus.append(W_yu)
            
            # For b_ys - uniform initialization for bias with k based on input dimension
            b_y = nn.Parameter(torch.zeros(self.ydim, device=self.device))
            nn.init.uniform_(b_y, -math.sqrt(k_y), math.sqrt(k_y))
            self.b_ys.append(b_y)
            
            # For W_us - uniform initialization with k based on input dimension (udim)
            W_u = nn.Parameter(torch.zeros(odim, udim, device=self.device))
            k_u = 1.0 / udim
            nn.init.uniform_(W_u, -math.sqrt(k_u), math.sqrt(k_u))
            self.W_us.append(W_u)
            
            # For b_us - uniform initialization for bias with k based on input dimension
            b_u = nn.Parameter(torch.zeros(odim, device=self.device))
            k_bu = 1.0 / odim  # For bias, using the output dimension as reference
            nn.init.uniform_(b_u, -math.sqrt(k_bu), math.sqrt(k_bu))
            self.b_us.append(b_u)

    # implements matrix vector multiplication between a matrix and a batch of vectors
    # (n x m) x (batch_size x m) = (batch_size, n)
    def mv(self, mat, vector_batch):
        return torch.t(mat @ torch.t(vector_batch))

    # the forward pass takes in a 2d tensor x of size (batch_size, x_size) and a 2d tensor y of size (batch_size, y_size)
    def forward(self, x, y):
        u = x.clone()
        z = y.clone()
        for i in range(len(self.Ws)):
            z = self.mv(
                self.W_zs[i], (z * torch.relu(self.mv(self.W_zus[i], u) + self.b_zs[i]))
            )
            z += self.mv(self.W_ys[i], (y * (self.mv(self.W_yus[i], u) + self.b_ys[i])))
            # try:
            #     z += self.mv(self.W_us[i], u) + self.b_us[i]
            # except:
            #     print(self.mv(torch.rand(self.W_us[i].shape), torch.rand(u.shape)))
            z += self.mv(self.W_us[i], u) + self.b_us[i]
            z = self.act(z)
            u = self.mv(self.Ws[i], u) + self.bs[i]
            u = self.act(u)

        z = self.mv(
            self.W_zs[-1], (z * torch.relu(self.mv(self.W_zus[-1], u) + self.b_zs[-1]))
        )
        z += self.mv(self.W_ys[-1], (y * (self.mv(self.W_yus[-1], u) + self.b_ys[-1])))
        z += self.mv(self.W_us[-1], u) + self.b_us[-1]
        return z

    def clamp_weights(self):
        with torch.no_grad():
            for layer in self.W_zs:
                layer.data = torch.clamp(layer.data, 0, torch.inf)

    def loss(self, x, y, label):
        return torch.sum((self.forward(x, y) - label) ** 2)

class PartialSubmodularMonotoneNet(nn.Module):
    def __init__(self, phi_layers, lamb, m_layers, u_layers, z_layers):
        super(PartialSubmodularMonotoneNet, self).__init__()

        self.sub = MonotoneSubmodularNet(phi_layers, lamb, m_layers)
        self.last = PartialInputConcaveNN(u_layers, z_layers)
    
    # is submodular over only the y inputs
    def forward(self, x, y):
        return self.last(x, self.sub(y))
    
    def clamp_weights(self):
        self.sub.clamp_weights()
        self.last.clamp_weights()

class DQN:
    """
    Args:
        layers (list[int]): represents the layers of the Q-network
            first value should be the dimension of the state, last value should be the dimension of the action
        env (string): specifies the environment the DQN is trained on
        experience_replay_capacity (int): the maximum capacity of the replay memory
        lr (float): learning rate for the optimizer
        gamma (float): discount factor for the reward function
        tau (float): the update rate of the target network (which is used to optimize the online network)
        eps (float): probability of choosing a random action as per the epsilon-greedy strategy
        device (string): device to train on
    """

    def __init__(
        self,
        phi_layers,
        lamb, 
        m_layers, 
        u_layers, 
        z_layers,
        action_space,
        experience_replay_capacity=10000,
        lr=0.001,
        gamma=0.1,
        tau=0.01,
        eps=1.0,
        eps_decay_amt=0.002, # for linearly decaying epsilon
        device="cpu",
    ):
        self.online_net = PartialSubmodularMonotoneNet(phi_layers, lamb, m_layers, u_layers, z_layers)
        self.target_net = PartialSubmodularMonotoneNet(phi_layers, lamb, m_layers, u_layers, z_layers)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.replay_memory = ReplayMemory(experience_replay_capacity, device)

        self.gamma = gamma
        self.tau = tau
        self.eps = eps
        self.eps_decay_amt = eps_decay_amt

        self.action_space = action_space

        self.update_target(
            soft=False
        )  # initially load the target net with the parameters of the online net

        self.device = device

    # choose a uniformly random action
    def random_action(self):
        return torch.tensor(
            self.action_space.sample(),
            dtype=torch.float32,
            requires_grad=True,
        )

    # selects an action according to the epsilon-greedy strategy 
    def select_action(self, state):
        if np.random.rand() < self.eps:
            return self.random_action()
        else:
            state = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                q_values = self.online_net(state)

            return torch.argmax(q_values)

    def train_step(self, batch_size):
        """
        Perform one training step using a batch from experience replay.
        Double DQN logic is implemented here.
        """
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.replay_memory.sample(
            batch_size
        )

        # Convert to tensors
        # states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        # rewards = (
        #     torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # )
        # next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Get current Q values from online network
        q_values = self.online_net(states).gather(1, actions)

        # Double DQN: action selection from the online network
        next_actions = self.online_net(next_states).argmax(1).unsqueeze(1)

        # Double DQN: Q values from the target network, using actions chosen by the online network
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()

        # print(q_values.shape)
        # print(next_q_values.shape)
        # print(rewards.shape)

        # Compute target Q values
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss (Huber loss or MSE loss)
        loss = F.mse_loss(q_values, q_targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self, soft=True):
        """
        Update the target network.
        If soft, performs a soft update using tau. If not, it directly copies the online network.
        """
        if soft:
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * online_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())
