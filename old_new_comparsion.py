# Dataset 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torch

# Old Code
import argparse
import numpy
import torch
from old_code.model.hopfield.network import DeepHopfieldEnergy
from old_code.model.function.network import Network
from old_code.model.function.cost import SquaredError
from old_code.model.hopfield.minimizer import FixedPointMinimizer
from old_code.training.sgd import EquilibriumProp, Backprop, AugmentedFunction
from old_code.training.epoch import Trainer, Evaluator
from old_code.training.monitor import Monitor, Optimizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

# New Code
from model.fully_connected_network import FullyConnectedNetwork
from core.energy import HopfieldEnergy
from core.updater import FixedPointUpdater as FixedPointUpdaterNew
from training.runner import Runner
from training.cost import SquaredError as SquaredErrorNew
from training.equilibrium_propagation import EquilibriumProp as EquilibriumPropNew
from util.config import Config
import random as Random

# Dataset
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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training["batch_size"], shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.training["batch_size"], shuffle=True)

# Old Code 


# Layers and Layer shapes for the model
num_hidden_layers = 1
num_neurons = 1
model = 'dhn-'+str(num_hidden_layers)+'h_twomoons'
layer_shapes = []
layer_shapes.append((1,))
for i in range(num_hidden_layers): #TODO: implement layer adapter function 
    layer_shapes.append((num_neurons,))
layer_shapes.append((1,))

# Other params 
num_iterations_inference = 50
num_iterations_training = 20
nudging = 0.2
num_epochs = 25

# More params
weight_init_dist = 'xavier_uniform' # Need to implement function to easily allow replacement
weight_gains = [1.0] * (num_hidden_layers+1)
learning_rates_weights = list(np.linspace(0.2, 0.01, num_hidden_layers+1))
learning_rates_biases = list(np.linspace(0.2, 0.01, num_hidden_layers+1))

# Model and Energy function 


energy_fn = DeepHopfieldEnergy(layer_shapes, weight_gains, weight_init_dist)
device = torch.device('cpu')
energy_fn.set_device(device) # Implement in class

output_layer = energy_fn.layers()[-1] # Output layer may not always be the last layer
cost_fn = SquaredError(output_layer)

network = Network(energy_fn) # Unify energy_fn and network same thing 

params = energy_fn.params() # Implement in class
layers = energy_fn.layers() # Implement in class
free_layers = network.free_layers() # Implement in class
# Manually set weights and biases

augmented_fn = AugmentedFunction(energy_fn, cost_fn) # input only network apply check in case of custom 
energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers) # input only network apply check in case of custom
estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training) # netowk, energy_fn, cost_fn, energy_minimizer_training as input only. Can I abstract it out further ???
estimator.nudging = nudging # Implement in class
estimator.variant = 'centered' # Implement in class

energy_minimizer_training.num_iterations = num_iterations_training
energy_minimizer_training.mode = 'synchronous'

learning_rates = learning_rates_biases + learning_rates_weights
optimizer = Optimizer(energy_fn, cost_fn, learning_rates, 0, 0)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
self_params =  params + cost_fn.params() 



# New Code
config = Config('simple_config.json')
network_new = FullyConnectedNetwork(config)
energy_fn_new = HopfieldEnergy(config)
cost_fn_new = SquaredErrorNew(config)
updater_new = FixedPointUpdaterNew(network, energy_fn, cost_fn, config)
scheduler_new = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
W, B = network_new.weights, network_new.biases
with torch.no_grad():
        B[0] = params[0].state
        B[1] = params[1].state
        W[1,0] = params[2].state
        W[2,1] = params[3].state
# Manually set weights and biases 
optimizer_new = torch.optim.SGD([W, B], lr=0.001)
differentiator_new = EquilibriumPropNew(network_new, energy_fn_new, cost_fn_new, updater_new, config, optimizer)


# Expand without detaching/cloning so gradients can flow back:
W_expanded = W.unsqueeze(0).expand(config.training['batch_size'], *W.shape)
B_expanded = B.unsqueeze(0).expand(config.training['batch_size'], *B.shape)

# Combined dataloader for comparision
for x, y in train_loader:

            # Reset the gradients
            optimizer.zero_grad()
            optimizer_new.zero_grad()


            # Set the input and target
            S_new = network_new.set_input(x)
            network.set_input(x, reset=False) 

            # Compute Equilibrium
            S_new = updater_new.compute_equilibrium(S_new, W_expanded, B_expanded, y)
            energy_minimizer_training.compute_equilibrium()  
            cost_fn.set_target(y) 
            

            # Compute Gradients 
            weight_grads, bias_grads, S_first, S_second = differentiator_new.compute_gradient(S_new, W_expanded, B_expanded, y)
            grads, layers_first, layers_second = estimator.compute_gradient()

            # Assign gradients to the original parameters. 
            W.grad, B.grad = weight_grads, bias_grads
            for param, grad in zip(params, grads): param.state.grad = grad 

            # Perform optimizer step 
            optimizer.step() 
            optimizer_new.step()
        
        # Schedule the learning rate
scheduler.step()
scheduler_new.step()

