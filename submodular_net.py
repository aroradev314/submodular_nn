import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from dqn import MonotoneSubmodularNet, concavity_regularizer
import wandb
import argparse
from torch.optim.lr_scheduler import StepLR

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
num_epochs = 200
batch_size = 1

m_layers = 2
phi_layers = [1, 50, 50, 50, 1]
lamb = 0.5

wandb.log({"M layers": m_layers, "Phi layers": phi_layers, "Lambda": lamb})

model = MonotoneSubmodularNet(phi_layers=phi_layers, lamb=lamb, m_layers=m_layers)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=10, gamma=3e-3)
criterion = nn.MSELoss()


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)
        reg_loss = concavity_regularizer(model.phi, strength=0.1*epoch)
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer.step()

        # if epoch == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None and param.grad.sum() > 0:
        #             print(f"Parameter: {name}, Gradient: {param.grad}")
        #             print(f"Parameter val: {param}")

        running_loss += total_loss.item() * batch_X.size(0)
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

model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        print(batch_X, outputs)
        print("actual:", batch_y)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)
test_loss = test_loss / len(test_dataset)

def check_integer_DR_submodularity(f_model, n=10, num_trials=1000, max_val=10, device='cpu'):
    """
    Empirically verify discrete DR-submodularity of a function over nonnegative integer vectors.

    Args:
        f_model: torch.nn.Module taking (batch_size, n) input and returning scalar output
        n: input dimensionality
        num_trials: number of random trials
        max_val: max integer value for components in vectors
    Returns:
        Fraction of trials where the DR condition holds
    """
    passed = 0

    for _ in range(num_trials):
        a = torch.randint(0, max_val, (n,), device=device) 
        delta = torch.randint(0, max_val, (n,), device=device)
        b = a + delta  # ensures b >= a coordinatewise

        i = torch.randint(0, n, (1,)).item()
        ei = torch.zeros(n, dtype=torch.long, device=device)
        ei[i] = 1

        # vector -> number
        # a: [2, 4, 1, 5]
        # delta: [1, 0, 3, 4]
        # b: [3, 4, 4, 9]
        # x: [0, 0, 1, 0]

        # f(b + x) - f(b) <= f(a + x) - f(a)

        with torch.no_grad():
            f_a = f_model(a.unsqueeze(0).float()).item()
            f_a_plus = f_model((a + ei).unsqueeze(0).float()).item()
            f_b = f_model(b.unsqueeze(0).float()).item()
            f_b_plus = f_model((b + ei).unsqueeze(0).float()).item()

        delta_a = f_a_plus - f_a
        delta_b = f_b_plus - f_b

        if delta_a >= delta_b - 1e-5:  # tolerance for float comparison
            passed += 1

    print("passed: ", passed)
    print("num trials", num_trials)
    return passed / num_trials


valid = check_integer_DR_submodularity(model, 5)
print("percent submodular:", valid)