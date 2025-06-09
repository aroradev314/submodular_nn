from dqn import DQN
from replay_memory import ReplayMemory
from prize_grid_env import GridWorld, GridWorldGym
import yaml
import argparse
import os
from utils.visualization import Visu
import pickle
import sys
import torch
import wandb

workspace = os.getcwd()
print(workspace)
print(sys.path)

wandb.init(project="submodular-nn")

parser = argparse.ArgumentParser(description='A foo that bars')
parser.add_argument('-param', default="subrl")  # params

parser.add_argument('-env', type=int, default=1)
parser.add_argument('-i', type=int, default=8)  # initialized at origin
args = parser.parse_args()

with open(os.path.join(workspace, "params", args.param + ".yaml")) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
print(params)

horizon = params["env"]["horizon"] # horizon
epochs = params["alg"]["epochs"]

env_load_path = workspace + \
    "/environments/" + params["env"]["node_weight"]+ "/env_" + \
    str(args.env)

environment = GridWorldGym(
    env_params=params["env"], common_params=params["common"], visu_params=params["visu"], alg_params=params["alg"], env_file_path=env_load_path)

if params["env"]["node_weight"] == "entropy" or params["env"]["node_weight"] == "steiner_covering" or params["env"]["node_weight"] == "GP": 
    a_file = open(env_load_path +".pkl", "rb")
    data = pickle.load(a_file)
    a_file.close()

if params["env"]["node_weight"] == "entropy":
    environment.env.cov = data
if params["env"]["node_weight"] == "steiner_covering":
    environment.env.items_loc = data
if params["env"]["node_weight"] == "GP":
    environment.env.weight = data

visu = Visu(env_params=params["env"])

agent = DQN([6, 10, 10, 1], 0.5, 10, [3, 100, 100, 1], [6, 10, 10, 10, 1], environment.action_space, eps_decay_amt=0.003, experience_replay_capacity=1000)
horizon = 20

# the batch size is 500 thats why it is being individually added to each 1

BATCH_SIZE = 100

glob_traj = []
for episode in range(epochs):
    environment = GridWorldGym(
    env_params=params["env"], common_params=params["common"], visu_params=params["visu"], alg_params=params["alg"], env_file_path=env_load_path)

    if params["env"]["node_weight"] == "entropy" or params["env"]["node_weight"] == "steiner_covering" or params["env"]["node_weight"] == "GP": 
        a_file = open(env_load_path +".pkl", "rb")
        data = pickle.load(a_file)
        a_file.close()

    if params["env"]["node_weight"] == "entropy":
        environment.env.cov = data
    if params["env"]["node_weight"] == "steiner_covering":
        environment.env.items_loc = data
    if params["env"]["node_weight"] == "GP":
        environment.env.weight = data

    state = (environment.reset().squeeze(), environment.env.get_prize_cnt(environment.mat_state))
    total_reward = 0
    total_loss = 0
    # agent.eps = 0

    traj = []
    print("started")
    for t in range(1, horizon):
        traj.append(state)
        action = agent.select_action(state)
        next_state, reward, done, info = environment.step(action)
        next_state = next_state.squeeze()

        agent.replay_memory.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done: 
            break

        if len(agent.replay_memory) > BATCH_SIZE:
            loss = agent.train_step(BATCH_SIZE)
            total_loss += loss
        
        agent.update_target() # do a soft update of the target network

    agent.eps = max(0.1, agent.eps - agent.eps_decay_amt)
    print(total_reward)
    print(total_loss)

    wandb.log({"Episode": episode, 
               "Total Reward": total_reward, 
               "Average Loss": total_loss})


