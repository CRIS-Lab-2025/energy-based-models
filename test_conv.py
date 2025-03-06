import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms

# Import our custom modules
from model.convolutional_network import ConvolutionalNetwork
from core.energy import ConvHopfieldEnergy
from core.updater import FixedPointUpdater
from training.runner import Runner
from training.cost import SquaredError
from training.equilibrium_propagation import EquilibriumProp
from util.config import Config

# Create a custom config for convolutional network
config = Config()
# Set a local path for results to avoid permission errors
config.path = "./results_conv"
config.device = "cpu"  # Change to "cuda" if you have a GPU
config.training["wandb"] = False
config.training["tensorboard"] = False
config.training["save_model"] = False
config.training["log"] = False
config.training["batch_size"] = 4
config.training["num_epochs"] = 1

# Set model parameters for convolutional network
config.model["channels"] = [1, 8, 16, 1]  # Input, hidden, output channels
config.model["kernel_size"] = 3
config.model["input_shape"] = (1, 28, 28)  # MNIST image shape
config.model["activation"] = "hard-sigmoid"

# Set gradient propagation parameters
config.gradient_propagation = {
    "name": "equilibrium_propagation",
    "nudging": 0.1,
    "variant": "centered",
    "use_alternative_formula": False
}

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use a small subset of MNIST for testing
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
train_dataset = torch.utils.data.Subset(train_dataset, range(1000))

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
test_dataset = torch.utils.data.Subset(test_dataset, range(200))

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=config.training["batch_size"], 
    shuffle=True, 
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=config.training["batch_size"], 
    shuffle=False
)

# Initialize the network and energy function
network = ConvolutionalNetwork(config)
energy_fn = ConvHopfieldEnergy(config)
cost_fn = SquaredError(config)
updater = FixedPointUpdater(network, energy_fn, cost_fn, config)

# Get weights and biases
W, B = network.weights, network.biases

# Initialize optimizer with a lower learning rate
optimizer = torch.optim.SGD([W, B], lr=0.0001)

# Initialize the differentiator
differentiator = EquilibriumProp(network, energy_fn, cost_fn, updater, config, optimizer)

# Create the runner
runner = Runner(config, network, train_loader, differentiator, updater, optimizer, inference_dataloader=test_loader)

# Run training
print("Starting training of convolutional network...")
runner.run_training()

print("Training complete!") 