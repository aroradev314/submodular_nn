# Installation

To install the Conda environment, run 
```
conda env create -f submodular_nn.yml
```

Then, activate it: 
```
conda activate submodular_nn
```

# Usage

To train the network, run
```
python3 submodular_net.py --modular
```
if you want to train using a modular reward function, or 

```
python3 submodular_net.py --no-modular
```
if you want to train using a submodular reward function.

We use `wandb` for visualization, so you will be prompted to login to your account. 

