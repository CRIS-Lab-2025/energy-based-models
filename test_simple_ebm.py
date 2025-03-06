import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

# Simple energy-based model components

class SimpleHopfieldEnergy:
    """A simplified Hopfield energy function for demonstration purposes."""
    
    def __init__(self):
        pass
    
    def energy(self, state, weight, bias):
        """Compute the energy of the state given weights and biases."""
        # E(S) = -0.5 * S^T W S - S^T b
        batch_size = state.shape[0]
        energy = torch.zeros(batch_size, device=state.device)
        
        # Compute quadratic term
        for b in range(batch_size):
            s = state[b].view(-1)  # Flatten the state
            w = weight[b] if weight.dim() > 2 else weight  # Handle batched weights
            energy[b] = -0.5 * torch.dot(s, torch.mv(w, s))
            
            # Add bias term
            if bias is not None:
                energy[b] -= torch.dot(s, bias)
        
        return energy
    
    def gradient(self, state, weight, bias):
        """Compute the gradient of the energy with respect to the state."""
        # Gradient = -W*S - b
        batch_size = state.shape[0]
        state_flat = state.view(batch_size, -1)
        grad = torch.zeros_like(state_flat)
        
        for b in range(batch_size):
            s = state_flat[b]
            w = weight[b] if weight.dim() > 2 else weight
            grad[b] = -torch.mv(w, s)
            
            if bias is not None:
                grad[b] -= bias
        
        return grad.view_as(state)

class SimpleEBMNetwork(nn.Module):
    """A simplified energy-based model network."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleEBMNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_size = input_size + hidden_size + output_size
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.zeros(self.total_size, self.total_size))
        self.bias = nn.Parameter(torch.zeros(self.total_size))
        
        # Initialize state
        self.state = None
        
        # Define layers for easier access
        self.input_layer = list(range(0, input_size))
        self.hidden_layer = list(range(input_size, input_size + hidden_size))
        self.output_layer = list(range(input_size + hidden_size, self.total_size))
        
        # Define free layers (all except input)
        self.free_layers = self.hidden_layer + self.output_layer
        
        # Initialize weights with small random values
        nn.init.xavier_uniform_(self.weight)
        
        # Make the weight matrix symmetric
        with torch.no_grad():
            weight_data = 0.5 * (self.weight.data + self.weight.data.t())
            self.weight.data = weight_data
            
            # Zero out the diagonal
            self.weight.data.fill_diagonal_(0)
    
    def set_input(self, x):
        """Set the input layer of the network."""
        batch_size = x.shape[0]
        
        # Initialize state if not already done
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.total_size, device=x.device)
        
        # Flatten input if needed
        x_flat = x.view(batch_size, -1)
        
        # Set input layer
        self.state[:, :self.input_size] = x_flat
        
        # Initialize hidden and output layers with small random values
        self.state[:, self.input_size:] = torch.rand(batch_size, self.hidden_size + self.output_size, 
                                                    device=x.device) * 0.1
        
        return self.state
    
    def forward(self, x):
        """Forward pass through the network."""
        # Set input
        state = self.set_input(x)
        
        # Return the output layer
        return state[:, self.output_layer]

class SimpleFixedPointUpdater:
    """A simplified fixed point updater for energy-based models."""
    
    def __init__(self, network, energy_fn, iterations=20):
        self.network = network
        self.energy_fn = energy_fn
        self.iterations = iterations
    
    def step(self, state, weight, bias, target=None, nudging=0):
        """Perform one step of fixed point update."""
        # Compute energy gradient
        grad = self.energy_fn.gradient(state, weight, bias)
        
        # Add cost gradient if target is provided
        if target is not None and nudging != 0:
            # Simple MSE cost gradient for output layer
            output_indices = self.network.output_layer
            batch_size = state.shape[0]
            
            # Convert target to one-hot if it's a class index
            if target.dim() == 1:
                one_hot = torch.zeros(batch_size, len(output_indices), device=target.device)
                one_hot.scatter_(1, target.unsqueeze(1), 1)
                target = one_hot
            
            # Compute cost gradient
            output = state[:, output_indices]
            cost_grad = torch.zeros_like(state)
            cost_grad[:, output_indices] = output - target
            
            # Add to energy gradient
            grad = grad + nudging * cost_grad
        
        # Update state with gradient
        new_state = state.clone()
        
        # Only update free layers
        for idx in self.network.free_layers:
            new_state[:, idx] = torch.sigmoid(-grad[:, idx])
        
        return new_state
    
    def compute_equilibrium(self, state, weight, bias, target=None, nudging=0):
        """Compute the equilibrium state by iterating the fixed point update."""
        current_state = state.clone()
        
        for i in range(self.iterations):
            current_state = self.step(current_state, weight, bias, target, nudging)
        
        return current_state

class SimpleEquilibriumProp:
    """A simplified equilibrium propagation algorithm."""
    
    def __init__(self, network, energy_fn, updater, nudging=0.1, variant="centered"):
        self.network = network
        self.energy_fn = energy_fn
        self.updater = updater
        self.nudging = nudging
        self.variant = variant
        
        # Set nudging values based on variant
        if variant == "positive":
            self.first_nudging = 0
            self.second_nudging = nudging
        elif variant == "negative":
            self.first_nudging = -nudging
            self.second_nudging = 0
        elif variant == "centered":
            self.first_nudging = -nudging
            self.second_nudging = nudging
        else:
            raise ValueError("Invalid variant. Expected 'positive', 'negative', or 'centered'.")
    
    def compute_gradient(self, state, weight, bias, target):
        """Compute parameter gradients using equilibrium propagation."""
        # First phase: compute equilibrium with first nudging
        first_state = self.updater.compute_equilibrium(state, weight, bias, target, self.first_nudging)
        
        # Second phase: compute equilibrium with second nudging
        second_state = self.updater.compute_equilibrium(state, weight, bias, target, self.second_nudging)
        
        # Compute parameter gradients
        batch_size = state.shape[0]
        weight_grad = torch.zeros_like(weight)
        bias_grad = torch.zeros_like(bias)
        
        # Compute weight gradients
        for b in range(batch_size):
            s1 = first_state[b].view(-1)
            s2 = second_state[b].view(-1)
            
            # Outer product for weight gradient
            w_grad1 = torch.outer(s1, s1)
            w_grad2 = torch.outer(s2, s2)
            
            weight_grad += (w_grad2 - w_grad1) / (self.second_nudging - self.first_nudging) / batch_size
        
        # Compute bias gradients
        bias_grad = (second_state.mean(dim=0) - first_state.mean(dim=0)) / (self.second_nudging - self.first_nudging)
        
        return weight_grad, bias_grad

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
train_dataset = torch.utils.data.Subset(train_dataset, range(500))

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
test_dataset = torch.utils.data.Subset(test_dataset, range(100))

# Create data loaders
batch_size = 16
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    drop_last=True
)

# Initialize components
input_size = 28 * 28  # MNIST image size
hidden_size = 100
output_size = 10  # 10 classes for MNIST

network = SimpleEBMNetwork(input_size, hidden_size, output_size)
energy_fn = SimpleHopfieldEnergy()
updater = SimpleFixedPointUpdater(network, energy_fn, iterations=10)
equilibrium_prop = SimpleEquilibriumProp(network, energy_fn, updater, nudging=0.1, variant="centered")

# Define optimizer
optimizer = optim.SGD([network.weight, network.bias], lr=0.01)

# Training function
def train(network, energy_fn, updater, equilibrium_prop, train_loader, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch_size = inputs.shape[0]
            
            # Set input
            state = network.set_input(inputs)
            
            # Compute parameter gradients
            weight_grad, bias_grad = equilibrium_prop.compute_gradient(
                state, network.weight, network.bias, labels
            )
            
            # Update parameters
            optimizer.zero_grad()
            network.weight.grad = weight_grad
            network.bias.grad = bias_grad
            optimizer.step()
            
            # Compute final state for evaluation
            final_state = updater.compute_equilibrium(state, network.weight, network.bias)
            
            # Compute output
            output = final_state[:, network.output_layer]
            
            # Compute accuracy
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i+1) % 5 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Accuracy: {100 * correct/total:.2f}%")
        
        print(f"Epoch {epoch+1}, Accuracy: {100 * correct/total:.2f}%")

# Testing function
def test(network, energy_fn, updater, test_loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            # Set input
            state = network.set_input(inputs)
            
            # Compute equilibrium
            final_state = updater.compute_equilibrium(state, network.weight, network.bias)
            
            # Compute output
            output = final_state[:, network.output_layer]
            
            # Compute accuracy
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Run training
print("Starting training...")
train(network, energy_fn, updater, equilibrium_prop, train_loader, optimizer, num_epochs=1)

# Run testing
print("Testing the model...")
test_accuracy = test(network, energy_fn, updater, test_loader)

print("Training and testing complete!")
print(f"Final test accuracy: {test_accuracy:.2f}%") 