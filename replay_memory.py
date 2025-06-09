import random
import torch
from collections import deque   

class ReplayMemory:
    def __init__(self, capacity, dev):
        self.buffer = deque(maxlen=capacity)
        self.device = dev
    
    def add(self, state, action, reward, next_state, done):
        experience = (torch.tensor(state).float().to(self.device), action.float().to(self.device), 
                      torch.tensor(reward).to(self.device), torch.tensor(next_state).float().to(self.device), 
                      torch.tensor(done).to(self.device))
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size, "not enough samples!"
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
