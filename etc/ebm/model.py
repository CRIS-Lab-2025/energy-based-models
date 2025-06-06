import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import pickle
from util.activation import *
from util.energy import *

class Network:
    """
    Implements a multi-layer neural network trained using the Equilibrium Propagation (EP)
    learning algorithm, based on Scellier et al. (2017) "Equilibrium Propagation: Bridging
    the Gap Between Energy-Based Models and Backpropagation".

    The network dynamics relax towards minimizing an energy function (e.g., Hopfield energy).
    Learning involves two phases:
    1. Negative Phase: The network settles to a free fixed point (s^0) given an input.
    2. Positive Phase: The output layer is nudged towards a target, and the network
       settles to a weakly-clamped fixed point (s^beta).
    The parameter updates approximate gradient descent on a cost function using the
    difference between these two phases.

    This process can also be viewed through the lens of variational EM (Bengio et al., 2015,
    "Towards Biologically Plausible Deep Learning"), where the negative phase performs
    approximate inference (E-step) and the positive phase + update performs the parameter
    update (M-step).
    """
    def __init__(self, name, external_world, hyperparameters={}):
        self.path = name + ".save"
        self.external_world = external_world
        self.hyperparameters = hyperparameters

        input_size = external_world.x.shape[1]
        output_size = hyperparameters.get("output_size", len(torch.unique(external_world.y)))
        layer_sizes = [input_size] + hyperparameters["hidden_sizes"] + [output_size]
        self.clamped_layers = [0]

        self.biases, self.weights, self.training_curves = self._initialize_params(layer_sizes)
        self.batch_size = hyperparameters["batch_size"]
        self.dataset_size = external_world.size_dataset
        self.persistent_particles = [torch.zeros((self.dataset_size, size)) for size in layer_sizes[1:]]
        self.index = 0
        self.grads = []

    def _initialize_params(self, layer_sizes):
        biases = [torch.zeros(size) for size in layer_sizes]
        weights = [torch.tensor(np.random.uniform(-np.sqrt(6 / (n_in + n_out)), 
                        np.sqrt(6 / (n_in + n_out)), (n_in, n_out)), dtype=torch.float32)
                   for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        return biases, weights, {"training error": [], "validation error": []}

    def update_mini_batch_index(self, index):
        start, end = index * self.batch_size, (index + 1) * self.batch_size
        self.x_data, self.y_data = self.external_world.x[start:end], self.external_world.y[start:end]
        #print(self.x_data.shape)
        output_size = self.hyperparameters["output_size"]
        self.y_data_one_hot = F.one_hot(self.y_data, num_classes=output_size).float()
        self.layers = [self.x_data] + [p[start:end] for p in self.persistent_particles]

    def energy(self, layers):
        """
        Compute the energy function E for the current layers.
        Refers to Eq. 1 in the EP paper (Scellier et al., 2017) for the Hopfield energy.
        E(u) = (1/2) * sum(u_i^2) - (1/2) * sum(W_ij * rho(u_i) * rho(u_j)) - sum(b_i * rho(u_i))
        """
        energy_fn = self.hyperparameters["energy_fn"] if "energy_fn" in self.hyperparameters else "hopfield"

        if energy_fn == 'none':
            pass
        elif energy_fn == 'hopfield':
            return hopfield(layers, self.weights, self.biases)
        else:
            raise ValueError('Unknown energy function type: {}'.format(energy_fn))

    def cost(self, layers):
        """
        Compute the cost function C, measuring the discrepancy between the network's
        output layer and the target labels.
        Refers to Eq. 2 in the EP paper (Scellier et al., 2017).
        C = (1/2) * ||y - target||^2
        where y is the state of the output layer.
        """
        # Squared error cost between the last layer and the one-hot labels.
        return ((layers[-1] - self.y_data_one_hot) ** 2).sum(dim=1)
    
    def activation(self, neurons):
        """Compute the activation of the given neurons' values."""
        activation = self.hyperparameters["activation"] if "activation" in self.hyperparameters else "pi"
        return get_activation(activation, neurons)
    
    def measure(self):
        """Measure the average energy, cost, and error over the current mini-batch."""
        E = self.energy(self.layers).mean().item()
        C = self.cost(self.layers).mean().item()
        y_prediction = self.layers[-1].argmax(dim=1)
        error = (y_prediction != self.y_data).float().mean().item()
        return E, C, error
    
    def forward(self, dataloader, n_iterations):
        """
        Perform forward pass with negative phase relaxation on a given dataloader.
        This primarily runs the negative phase to evaluate the model's predictions
        and associated energy/cost/error without performing parameter updates.
        """
        total_energy = 0
        total_cost = 0
        total_error = 0
        n_batches = 0
        
        # Store original persistent particles
        original_particles = [p.clone() for p in self.persistent_particles]
        
        x,y = dataloader
        n_batches = x.shape[0]
        # Move data to device and prepare one-hot labels
        self.x_data = x
        self.y_data = y
        self.y_data_one_hot = F.one_hot(self.y_data, num_classes=self.weights[-1].shape[1]).float()
        
        # Initialize layers for this batch
        print(self.x_data.shape)
        self.layers = [self.x_data] + [p[:x.size(0)] for p in self.persistent_particles]
        
        # Perform negative phase
        self.negative_phase(n_iterations)
        
        # Measure metrics
        E, C, error = self.measure()
        total_energy += E
        total_cost += C
        total_error += error

        
        # Restore original persistent particles
        self.persistent_particles = original_particles
        
        # Return average metrics
        return (total_energy / n_batches, 
                total_cost / n_batches, 
                total_error / n_batches)

    def negative_phase(self, n_iterations):
        """
        Perform the negative phase relaxation (free phase).

        The network state evolves according to gradient dynamics on the energy E
        (equivalent to dynamics on F = E + beta*C with beta=0).
        Refers to Eq. 4/7 in the EP paper (Scellier et al., 2017): ds/dt = -dF/ds
        This discrete-time implementation iteratively updates layer states to reach
        a fixed point s^0 where dF/ds = 0.
        The input layer (layers[0]) is clamped.
        """
        # Copy the current mini-batch layers to iterate on.
        current_layers = [layer.clone() for layer in self.layers]
        for _ in range(n_iterations):
            new_layers = [torch.zeros(i.shape) for i in self.layers]  # input layer remains clamped.
            new_layers[0] = current_layers[0]
            # For hidden layers (except the final output layer).
            # shuffle range(1, len(self.layers) - 1)
            iter_order = range(1, len(self.layers) - 1)
            for k in iter_order:
                hidden_input = (torch.matmul(new_layers[k-1], self.weights[k - 1]) +
                                torch.matmul(current_layers[k + 1], self.weights[k].t()) +
                                self.biases[k])
                # Batch normalization
                hidden_input = self.activation(hidden_input)
                # Min-max normalization to [0,1] range
                min_val = hidden_input.min(dim=0)[0]
                max_val = hidden_input.max(dim=0)[0]
                hidden_input = (hidden_input - min_val) / (max_val - min_val + 1e-5)  # Add epsilon to avoid division by zero
                #print(f"hidden_input mean: {hidden_input.mean():.4f}, std: {hidden_input.std():.4f}, max: {hidden_input.max():.4f}, min: {hidden_input.min():.4f}")
                # Batch normalization using mean and std
                # mean = hidden_input.mean(dim=0)
                # std = hidden_input.std(dim=0) + 1e-5
                # hidden_input = (hidden_input - mean) / std
                hidden_input = hidden_input - hidden_input.mean(dim=0)
                #print(f"hidden_input mean: {hidden_input.mean():.4f}, std: {hidden_input.std():.4f}, max: {hidden_input.max():.4f}, min: {hidden_input.min():.4f}")
                new_layers[k]=(hidden_input)
            # Compute output layer.3:45pm
            output_input = torch.matmul(new_layers[-2], self.weights[-1]) + self.biases[-1]
            new_layers[-1] = self.activation(output_input)
            current_layers = new_layers
        # Update the persistent particles for the current mini-batch.
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        for i in range(len(self.persistent_particles)):
            self.persistent_particles[i][start:end] = current_layers[i + 1].detach()
        self.layers = [self.x_data] + [p[start:end] for p in self.persistent_particles]
    
    def positive_phase(self, n_iterations, *alphas):
        """
        Perform the positive phase (nudged phase) and update parameters.

        1. Nudged Relaxation: The output layer is weakly clamped towards the target
           (self.y_data_one_hot replaces layers[-1]). The network state then evolves
           according to dynamics on the total energy F = E + beta*C (with beta > 0 implicitly).
           The system settles to a new fixed point s^beta near s^0.
           Refers to Sec 3.2, 3.3 in the EP paper (Scellier et al., 2017).

        2. Parameter Update: Weights (W) and biases (b) are updated based on the
           difference between the state derivatives in the positive (beta > 0) and
           negative (beta = 0) phases.
           Refers to Eq. 24 in the EP paper (Scellier et al., 2017):
           Delta theta proportional to -(1/beta) * (dF/dtheta(s^beta) - dF/dtheta(s^0))
           Here, the update is approximated using the difference in layer activations
           (Delta_layers) between the s^beta and s^0 states.
           The update rule resembles contrastive Hebbian learning.
        """
        batch_size = self.x_data.shape[0]
        # Initialize the backprop scan: all layers except last remain unchanged,
        # and the final layer is replaced by the clamped one-hot label (nudging).
        initial_layers = self.layers[:-1] + [self.y_data_one_hot]
        current_layers = [layer.clone() for layer in initial_layers]
        for _ in range(n_iterations):
            new_layers = [current_layers[-1]]  # start with the top layer.
            # Backpropagate from the top hidden layer to the second layer.
            for k in range(len(self.layers) - 2, 0, -1):
                back_input = (torch.matmul(self.layers[k - 1], self.weights[k - 1]) +
                              torch.matmul(new_layers[-1], self.weights[k].t()) +
                              self.biases[k])
                back_input = self.activation(back_input)
                # Min-max normalization to [0,1] range
                min_val = back_input.min(dim=0)[0]
                max_val = back_input.max(dim=0)[0]
                back_input = (back_input - min_val) / (max_val - min_val + 1e-5)  # Add epsilon to avoid division by zero
                # back_input = back_input - 0.5 / 0.5
                back_input = back_input - back_input.mean(dim=0)
                # Batch normalization using mean and std
                # mean = back_input.mean(dim=0)
                # std = back_input.std(dim=0) + 1e-5
                # back_input = (back_input - mean) / std
                #print(f"back_input mean: {back_input.mean():.4f}, std: {back_input.std():.4f}, max: {back_input.max():.4f}, min: {back_input.min():.4f}")
                new_layers.append(back_input)
            new_layers.append(self.layers[0])
            new_layers.reverse()
            current_layers = new_layers
        # Compute the difference (Delta) between the new layers (s^beta state) and the persistent ones (s^0 state).
        Delta_layers = [new - old for new, old in zip(current_layers[1:], self.layers[1:])]
        #print(f"Delta_layers mean: {Delta_layers[0].mean():.4f}, std: {Delta_layers[0].std():.4f}, max: {Delta_layers[0].max():.4f}, min: {Delta_layers[0].min():.4f}")
        # Update biases using the difference (approximating EP gradient Eq. 24 for biases).
        for i, delta in enumerate(Delta_layers, start=1):
            self.biases[i] = self.biases[i] + alphas[i - 1] * delta.mean(dim=0)
        #
        # Update weights using the difference (approximating EP gradient Eq. 24 for weights).
        # This uses a Hebbian-like update: outer product of pre-synaptic state (self.layers[i])
        # and the change in post-synaptic state (delta).
        for i, delta in enumerate(Delta_layers):
            self.weights[i] = (self.weights[i] + alphas[i] * (self.layers[i].t() @ delta) / batch_size )
            grads = self.layers[i].t()@delta
            #print(f"grads mean: {grads.mean():.4f}, std: {grads.std():.4f}, max: {grads.max():.4f}, min: {grads.min():.4f}")
        
    def backward(self,output, n_iterations=10,clamped_layers=[-1]):
        """
        Perform backward relaxation dynamics, primarily used for generative tasks
        (e.g., reconstructing input from a given output).
        Clamps specified layers (defaulting to the output layer) and lets the
        network settle to infer the state of other layers (e.g., the input layer).
        Similar relaxation dynamics as negative_phase but potentially clamping
        different layers and propagating influence 'backward'.
        """
        # Copy the current mini-batch layers to iterate on.
        current_layers = [layer.clone() for layer in self.layers]
        current_layers[-1] = output
        for _ in range(n_iterations):
            new_layers = [torch.zeros(i.shape) for i in self.layers]  # input layer remains clamped.
            for i in clamped_layers: new_layers[i] = current_layers[i]
            # For hidden layers (except the final output layer).
            # shuffle range(1, len(self.layers) - 1)
            iter_order = np.random.permutation(range(1, len(self.layers) - 1))
            for k in iter_order:
                hidden_input = (torch.matmul(new_layers[k-1], self.weights[k - 1]) +
                                torch.matmul(current_layers[k + 1], self.weights[k].t()) +
                                self.biases[k])
                new_layers[k]=(self.activation(hidden_input))
                # Batch normalization using mean and std
                # mean = new_layers[k].mean(dim=0)
                # std = new_layers[k].std(dim=0) + 1e-5
                # new_layers[k] = (new_layers[k] - mean) / std
                # Min-max normalization to [0,1] range
                min_val = new_layers[k].min(dim=0)[0]
                max_val = new_layers[k].max(dim=0)[0]
                new_layers[k] = (new_layers[k] - min_val) / (max_val - min_val + 1e-5)  # Add epsilon to avoid division by zero
                # new_layers[k] = new_layers[k] - new_layers[k].mean(dim=0)
            # Compute output layer.
            output_input = torch.matmul(new_layers[1], self.weights[0].T) + self.biases[0]
            new_layers[0] = self.activation(output_input)
            current_layers = new_layers
        # Update the persistent particles for the current mini-batch.
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        for i in range(len(self.persistent_particles)):
            self.persistent_particles[i][start:end] = current_layers[i + 1].detach()
        self.layers = [self.x_data] + [p[start:end] for p in self.persistent_particles]
        return current_layers[0]

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




