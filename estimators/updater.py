# class Updater(ABC):
#     # TODO
#     def __init__(self):
#         super().__init__()

#     @abstractmethod
#     def pre_activate():
#         pass

#     def step():
#         '''
#         Perform a step. Update weights and biases accordingly

#         TODO:
#         1. figure out some ordering of the layers (fwd/bkwd/sync/async)
#         2. For each layer (in order): Do pre-activation, then activation
#         '''
        
#         pass

from abc import ABC, abstractmethod
import torch

# =============================================================================
# Layer Updaters: abstract base and concrete implementations
# =============================================================================
class ParamUpdater:
    """
    Abstract class to update the state of a parameter
    
    The param theta is updated in proprtion to the gradient of a function E
    A central quantity used to determine the param's update is the gradient dE/dtheta

    Methods
    -------
    grad():
        Computes the gradient of the function wrt the param's state
    """

    def __init__(self, param, fn):
        """Creates an instance of ParamUpdater

        Args:
            param (Parameter): the param
            fn (Function): the function
        """

        self._param = param

        self.grad = fn.grad_param_fn(param)  # this is a method, not an attribute
        self.second_fn = fn.second_fn(param)  # this is a method, not an attribute

class LayerUpdater(ABC):
    """
    Abstract base class to update the state of a layer.

    Each updater has access to the layer and a function that computes the gradient 
    (via fn.grad_layer_fn(layer)).
    """
    def __init__(self, layer, fn):
        self.layer = layer
        self.grad = fn.grad_layer_fn(layer)  # function to compute dE/dz

    @abstractmethod
    def pre_activate(self):
        """
        Compute the pre-activation value for the layer.
        
        Returns:
            Tensor: pre-activation values (before applying the activation function)
        """
        pass

class GradientDescentUpdater(LayerUpdater):
    """
    Layer updater using gradient descent. The pre-activation is computed as a
    gradient descent step:
    
        new_state = current_state - step_size * grad
    """
    def __init__(self, layer, fn, step_size):
        super().__init__(layer, fn)
        self.step_size = step_size

    def pre_activate(self):
        return self.layer.state - self.step_size * self.grad()

class HopfieldUpdater(LayerUpdater):
    """
    Layer updater using Hopfield dynamics. The pre-activation is computed by taking
    the negative of the gradient:
    
        new_state = - grad
    """
    def pre_activate(self):
        return -self.grad()

# =============================================================================
# Unified Network State Updater
# =============================================================================

class NetworkStateUpdater:
    """
    Updates the network’s layers to reduce a given scalar function (e.g., an energy).
    
    Adjustable parameters (provided via the `config` dictionary):
        - algorithm: 'gradient' or 'hopfield'
        - step_size: (float) used only if algorithm=='gradient'
        - mode: update ordering ('forward', 'backward', 'synchronous', or 'asynchronous')
        - num_iterations: (int) number of iterations to run
    
    The updater works by:
      1. Determining an order in which to update the layers.
      2. For each layer: computing a pre-activation (using either gradient descent or 
         Hopfield dynamics) and then applying the layer’s activation function.
    """
    def __init__(self, fn, free_layers, config):
        """
        Args:
            fn: The function to minimize. This object must implement:
                - grad_layer_fn(layer): returns a function computing dE/dz
                - layers(): returns all network layers
                - params(): returns the network parameters (e.g., weights and biases)
            free_layers (list): Layers that will be updated.
            config (dict): Contains adjustable parameters:
                - algorithm: 'gradient' or 'hopfield'
                - step_size: (float, default=0.5) for gradient descent
                - mode: one of 'forward', 'backward', 'synchronous', 'asynchronous'
                - num_iterations: (int, default=15)
        """
        self.fn = fn
        self.all_layers = fn.layers()
        self.params = fn.params()
        self.config = config

        self.num_iterations = config.get('num_iterations', 15)
        self.mode = config.get('mode', 'asynchronous')
        self.algorithm = config.get('algorithm', 'gradient')
        self.step_size = config.get('step_size', 0.5)  # Only used for gradient descent

        # Create a list of layer updaters depending on the chosen algorithm.
        if self.algorithm == 'gradient':
            self.updaters = [
                GradientDescentUpdater(layer, fn, self.step_size)
                for layer in free_layers
            ]
        elif self.algorithm == 'hopfield':
            self.updaters = [
                HopfieldUpdater(layer, fn)
                for layer in free_layers
            ]
        else:
            raise ValueError("Invalid algorithm type: choose 'gradient' or 'hopfield'.")

        self._set_mode()

    def _set_mode(self):
        """Determine the update order for the layers based on the mode."""
        if self.mode == 'forward':
            # Update each layer one at a time in forward order.
            self.layer_groups = [[updater] for updater in self.updaters] * self.num_iterations
        elif self.mode == 'backward':
            # Update each layer one at a time in reverse order.
            self.layer_groups = [[updater] for updater in reversed(self.updaters)] * self.num_iterations
        elif self.mode == 'synchronous':
            # Update all layers at once.
            self.layer_groups = [self.updaters] * self.num_iterations
        elif self.mode == 'asynchronous':
            # Update layers in two alternating groups.
            self.layer_groups = [self.updaters[::2], self.updaters[1::2]] * self.num_iterations
        else:
            raise ValueError("Invalid mode: choose 'forward', 'backward', 'synchronous', or 'asynchronous'.")

    def step(self, layer_group):
        """
        Perform one update step on a group of layers.
        
        For each updater in the group:
          1. Compute the pre-activation value.
          2. Set the layer state to the pre-activation.
          3. Apply the layer's activation function.
        """
        for updater in layer_group:
            pre_activation = updater.pre_activate()
            updater.layer.state = pre_activation
            updater.layer.state = updater.layer.activate()

    def compute_equilibrium(self):
        """
        Run the update process for the configured number of iterations, then return
        the final states of all layers.
        
        Returns:
            dict: Keys are layer names; values are the final states.
        """
        for group in self.layer_groups:
            self.step(group)
        return {layer.name: layer.state for layer in self.all_layers}

    def compute_trajectory(self):
        """
        Record and return the trajectory of the layers (and parameters) throughout
        the update process.
        
        Returns:
            dict: Keys are layer and parameter names; values are lists of states over time.
        """
        trajectories = {layer.name: [layer.state.clone()] for layer in self.all_layers}
        trajectories.update({param.name: [param.state.clone()] for param in self.params})

        for group in self.layer_groups:
            # (Optional) Update parameters here if needed.
            for param in self.params:
                param.state = param.state + torch.zeros_like(param.state)
                trajectories[param.name].append(param.state.clone())
            self.step(group)
            for layer in self.all_layers:
                trajectories[layer.name].append(layer.state.clone())

        return trajectories

