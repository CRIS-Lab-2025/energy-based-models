import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from external_world import ExternalWorld

def pi(s):
    return torch.clamp(s, 0.0, 1.0)

class Network:
    def __init__(self, name, hyperparameters={}):
        self.path = name + ".save"
        self.biases, self.weights, self.hyperparameters, self.training_curves = self.__load_params(hyperparameters)
        
        # LOAD EXTERNAL WORLD (=DATA)
        self.external_world = ExternalWorld()
        
        # INITIALIZE PERSISTENT PARTICLES for each non-input layer
        dataset_size = self.external_world.size_dataset
        layer_sizes = [28 * 28] + self.hyperparameters["hidden_sizes"] + [10]
        self.persistent_particles = [
            torch.zeros((dataset_size, size), dtype=torch.float32)
            for size in layer_sizes[1:]
        ]
        
        self.batch_size = self.hyperparameters["batch_size"]
        self.index = 0
        self.update_mini_batch_index(self.index)
    
    def update_mini_batch_index(self, index):
        """Update the mini-batch index and slice the external data and persistent particles."""
        self.index = index
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        self.x_data = self.external_world.x[start:end]
        self.y_data = self.external_world.y[start:end]
        self.y_data_one_hot = F.one_hot(self.y_data, num_classes=10).float()
        # Layers: first layer is the clamped input, then one slice per persistent particle.
        self.layers = [self.x_data] + [p[start:end] for p in self.persistent_particles]
    
    def save_params(self):
        biases_values = [b.detach().numpy() for b in self.biases]
        weights_values = [W.detach().numpy() for W in self.weights]
        to_dump = (biases_values, weights_values, self.hyperparameters, self.training_curves)
        with open(self.path, "wb") as f:
            pickle.dump(to_dump, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __load_params(self, hyperparameters):
        hyper = hyperparameters.copy()
        if os.path.isfile(self.path):
            with open(self.path, "rb") as f:
                biases_values, weights_values, saved_hyper, training_curves = pickle.load(f)
            saved_hyper.update(hyper)
            hyper = saved_hyper
        else:
            layer_sizes = [28 * 28] + hyper["hidden_sizes"] + [10]
            biases_values = [np.zeros((size,), dtype=np.float32) for size in layer_sizes]
            weights_values = []
            # Glorot/Bengio weight initialization for each layer.
            for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = np.random.uniform(low=-limit, high=limit, size=(n_in, n_out)).astype(np.float32)
                weights_values.append(W)
            training_curves = {"training error": [], "validation error": []}
        
        # Convert loaded parameters to torch tensors.
        biases = [torch.tensor(b, dtype=torch.float32) for b in biases_values]
        weights = [torch.tensor(W, dtype=torch.float32) for W in weights_values]
        return biases, weights, hyper, training_curves
    
    def energy(self, layers):
        # Compute the energy function E for the current layers.
        # squared_norm: for each layer, sum of squares of each row, then sum over layers.
        squared_norm = sum([(layer * layer).sum(dim=1) for layer in layers]) / 2.0
        # linear_terms: for each layer, compute dot(layer, bias).
        linear_terms = -sum([torch.matmul(layer, b) for layer, b in zip(layers, self.biases)])
        # quadratic_terms: for each adjacent pair of layers.
        quadratic_terms = -sum([
            ((torch.matmul(pre, W)) * post).sum(dim=1)
            for pre, W, post in zip(layers[:-1], self.weights, layers[1:])
        ])
        return squared_norm + linear_terms + quadratic_terms
    
    def cost(self, layers):
        # Squared error cost between the last layer and the one-hot labels.
        return ((layers[-1] - self.y_data_one_hot) ** 2).sum(dim=1)
    
    def measure(self):
        """Measure the average energy, cost, and error over the current mini-batch."""
        E = self.energy(self.layers).mean().item()
        C = self.cost(self.layers).mean().item()
        y_prediction = self.layers[-1].argmax(dim=1)
        error = (y_prediction != self.y_data).float().mean().item()
        return E, C, error
    
    def negative_phase(self, n_iterations):
        """Perform the negative phase relaxation (forward pass)."""
        # Copy the current mini-batch layers to iterate on.
        current_layers = [layer.clone() for layer in self.layers]
        for _ in range(n_iterations):
            new_layers = [current_layers[0]]  # input layer remains clamped.
            # For hidden layers (except the final output layer).
            for k in range(1, len(self.layers) - 1):
                hidden_input = (torch.matmul(new_layers[-1], self.weights[k - 1]) +
                                torch.matmul(current_layers[k + 1], self.weights[k].t()) +
                                self.biases[k])
                new_layers.append(pi(hidden_input))
            # Compute output layer.
            output_input = torch.matmul(new_layers[-1], self.weights[-1]) + self.biases[-1]
            new_layers.append(pi(output_input))
            current_layers = new_layers
        # Update the persistent particles for the current mini-batch.
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        for i in range(len(self.persistent_particles)):
            self.persistent_particles[i][start:end] = current_layers[i + 1].detach()
        self.layers = [self.x_data] + [p[start:end] for p in self.persistent_particles]
    
    def positive_phase(self, n_iterations, *alphas):
        """Perform the positive phase (backprop-like relaxation and parameter update)."""
        batch_size = self.x_data.shape[0]
        # Initialize the backprop scan: all layers except last remain unchanged,
        # and the final layer is replaced by the clamped one-hot label.
        initial_layers = self.layers[:-1] + [self.y_data_one_hot]
        current_layers = [layer.clone() for layer in initial_layers]
        for _ in range(n_iterations):
            new_layers = [current_layers[-1]]  # start with the top layer.
            # Backpropagate from the top hidden layer to the second layer.
            for k in range(len(self.layers) - 2, 0, -1):
                back_input = (torch.matmul(self.layers[k - 1], self.weights[k - 1]) +
                              torch.matmul(new_layers[-1], self.weights[k].t()) +
                              self.biases[k])
                new_layers.append(pi(back_input))
            new_layers.append(self.layers[0])
            new_layers.reverse()
            current_layers = new_layers
        # Compute the difference (Delta) between the new layers and the persistent ones (skipping the input).
        Delta_layers = [new - old for new, old in zip(current_layers[1:], self.layers[1:])]
        # Update biases for layers 1 to end.
        for i, delta in enumerate(Delta_layers, start=1):
            self.biases[i] = self.biases[i] + alphas[i - 1] * delta.mean(dim=0)
        # Update weights for each connection.
        for i, delta in enumerate(Delta_layers):
            self.weights[i] = self.weights[i] + alphas[i] * (self.layers[i].t() @ delta) / batch_size
