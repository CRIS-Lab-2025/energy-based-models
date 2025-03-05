import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch
from model.fully_connected_network import FullyConnectedNetwork
from core.energy import HopfieldEnergy
from core.updater import FixedPointUpdater
from training.runner import Runner
from training.cost import SquaredError
from training.equilibrium_propagation import EquilibriumProp
from util.config import Config
import random as Random

# Create a toy dataset with 2 types of points. The first set of points is between 0 and 0.4 and the second set is between 0.6 and 1.
X_1 = np.random.uniform(0, 0.4, 100)
X_2 = np.random.uniform(0.6, 1, 100)
X_1 = torch.tensor(X_1).reshape(-1, 1)
X_2 = torch.tensor(X_2).reshape(-1, 1)
X = torch.cat([X_1, X_2])
Y = torch.cat([torch.zeros(100), torch.ones(100)],dim=0).long()



# Create a dataloader for the dataset
dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# Split the dataset into a training and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])




config = Config('simple_config.json')
network = FullyConnectedNetwork(config)
energy_fn = HopfieldEnergy(config)
cost_fn = SquaredError(config)
updater = FixedPointUpdater(network, energy_fn, cost_fn, config)
W, B = network.weights, network.biases
optimizer = torch.optim.SGD([W, B], lr=0.001)
differentiator = EquilibriumProp(network, energy_fn, cost_fn, updater, config, optimizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training["batch_size"], shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.training["batch_size"], shuffle=True)
runner = Runner(config, network, train_loader, differentiator, updater, optimizer, inference_dataloader=None)
runner.run_training()
