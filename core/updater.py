from abc import ABC, abstractmethod
import random
from core.activation import get_activation_neuron
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
    """
    Class used to #TODO
    """
    def __init__(self, network, energy_fn, cost_fn, config):
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
            layers = self.network.free_layers
            random.shuffle(layers)
            self.update_order = layers.append(self.network.free_layers[-1])
        else:
            raise ValueError("Invalid update order")

    def step(self, S, W, B, target=None, nudging=0):
        """TODO
        
        Args:
            S: state
            W: weights
            B: bias
        """
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
        

    def compute_equilibrium(self, S, W,B,target,nudging=0):
        """TODO"""
        # iterate for a fixed number of steps to reach equilibrium
        for i in range(self.iterations):
            S = self.step(S,W,B,target,nudging)

        return S




