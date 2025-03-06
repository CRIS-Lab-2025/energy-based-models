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




