from abc import ABC, abstractmethod
import random
from core.activation import get_activation_neuron
from util.config import Config
from core.energy import EnergyFunction
from training.cost import SquaredError
import torch

class Updater(ABC):
    # TODO
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _update_order(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def compute_equilibrium(self):
        pass


class FixedPointUpdater(Updater):
    """TODO: Header comment"""
    def __init__(self, network, energy_fn: EnergyFunction, cost_fn: SquaredError, config: Config):
        super().__init__()
        self.network = network
        self.energy_fn = energy_fn
        self.cost_fn = cost_fn
        self.config = config
        self.order = config.updater['training_order']
        self.iterations = config.updater['training_iterations']
        self._update_order()

    def _update_order(self):
        """TODO"""
        # determine the order in which to update the nodes
        self.update_order = []
        if self.order == 'synchronous':
            self.update_order = self.network.free_layers
        elif self.order == 'asynchronous':
            # randomize order
            layers = self.network.free_layers.copy()  # Create a copy to avoid modifying the original
            random.shuffle(layers)
            self.update_order = layers
        else:
            raise ValueError("Invalid update order")

    def step(self, S, W, B, target=None, nudging=0):
        """TODO"""
        # Check if we're dealing with a convolutional network
        if hasattr(self.network, '_conv_weights'):
            # For convolutional networks, we need to handle the update differently
            # We'll update the entire state tensor at once
            
            # Handle the case where S is a tuple (state, weights)
            if isinstance(S, tuple):
                S = S[0]
            
            # Get the current state
            current_state = S.clone()
            
            # Compute the energy gradient for the entire state
            state_grad, _ = self.energy_fn.full_gradient(current_state, W, B)
            
            # Add the cost gradient if target is provided
            if target is not None:
                cost_grad = self.cost_fn.gradient(current_state, target)
                state_grad += nudging * cost_grad
            
            # Update the state
            S = -state_grad
            
            # Apply activation function
            if self.network.activation == "sigmoid":
                S = torch.sigmoid(S)
            elif self.network.activation == "tanh":
                S = torch.tanh(S)
            elif self.network.activation == "relu":
                S = torch.nn.functional.relu(S)
            elif self.network.activation == "hard-sigmoid":
                S = torch.clamp(0.2 * S + 0.5, 0.0, 1.0)
                
            return S
        else:
            # Original implementation for fully connected networks
            for layer in self.update_order:
                for node in self.network.layers[layer]:
                    # compute the gradient for the node
                    grad, _ = self.energy_fn.node_gradient(S, W, B, node)
                    # add the cost gradient
                    if layer == self.cost_fn._layer and target is not None:
                            # index of node in the class layer
                            class_id = self.network.layers[self.cost_fn._layer].index(node)
                            cost_grad = self.cost_fn.node_gradient(grad, target, node, class_id)
                            grad += nudging * cost_grad
                    # update the state
                    S[:,node] = grad
                # layer activation
                S[:,self.network.layers[layer]] = get_activation_neuron(self.network.activation, S[:,self.network.layers[layer]])
            return S
        

    def compute_equilibrium(self, S, W, B, target, nudging=0):
        """TODO"""
        # iterate for a fixed number of steps to reach equilibrium
        print(f"Starting equilibrium computation with nudging={nudging}")
        for i in range(self.iterations):
            S = self.step(S,W,B,target,nudging)
            if i % 5 == 0:  # Print every 5 iterations
                print(f"  Iteration {i}: State mean={S.mean().item():.4f}, min={S.min().item():.4f}, max={S.max().item():.4f}")

        print(f"Finished equilibrium computation")
        return S




