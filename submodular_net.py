import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from dqn import MonotoneSubmodularNet
import wandb
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--modular", action=argparse.BooleanOptionalAction, help="Enable or disable modular mode")
args = parser.parse_args()

modular = args.modular
wandb.init(project="submodular_nn", config={"Modular": modular})

# Set random seed for reproducibility
np.random.seed(42)

# Number of prizes
n = 5

# Create a random d vector for n prizes (upper bounds on collection)
# For example, each d_i is drawn uniformly from 1 to 5
d = np.random.randint(1, 6, size=n)
print("d vec: ", d)

# Define the reward function
def reward_function(x, d):
    # x: array of collected amounts (shape: (n,))
    # d: array of upper bounds (shape: (n,))
    return np.sum(np.minimum(x, d))

def modular_reward_function(x, d):
    return np.sum(x)

# Function to generate a dataset
def generate_dataset(num_samples, d, extra=3, modular=False):
    # For each prize i, sample x_i uniformly from 0 to d[i] + extra
    X = np.zeros((num_samples, len(d)), dtype=int)
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        sample = [np.random.randint(0, d_i + extra + 1) for d_i in d]
        X[i, :] = sample
        if modular:
            y[i] = modular_reward_function(sample, d)
        else:
            y[i] = reward_function(sample, d)

    return X, y

# Generate training and testing sets
num_train = 100
num_test = 30

X_train, y_train = generate_dataset(num_train, d, modular=modular)
X_test, y_test = generate_dataset(num_test, d, modular=modular)

print("\nFirst 5 training samples (X_train and corresponding rewards):")
for i in range(5):
    print("X_train[{}] = {}, reward = {}".format(i, X_train[i], y_train[i]))

print("\nFirst 5 testing samples (X_test and corresponding rewards):")
for i in range(5):
    print("X_test[{}] = {}, reward = {}".format(i, X_test[i], y_test[i]))


# Hyperparameters
learning_rate = 1e-2
num_epochs = 100
batch_size = 1

model = MonotoneSubmodularNet([1, 20, 20, 20, 20, 1], 0.5, 2)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # if epoch == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None and param.grad.sum() > 0:
        #             print(f"Parameter: {name}, Gradient: {param.grad}")
        #             print(f"Parameter val: {param}")

        running_loss += loss.item() * batch_X.size(0)
        # model.clamp_weights()
    
    epoch_loss = running_loss / len(train_dataset)

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
    wandb.log({"epoch": epoch + 1, "Train Loss": epoch_loss, "Test Loss": test_loss})
    # time.sleep(2)

# model.eval()
# test_loss = 0.0
# with torch.no_grad():
#     for batch_X, batch_y in test_loader:
#         outputs = model(batch_X)
#         print(batch_X, outputs)
#         print("actual:", batch_y)
#         loss = criterion(outputs, batch_y)
#         test_loss += loss.item() * batch_X.size(0)
# test_loss = test_loss / len(test_dataset)