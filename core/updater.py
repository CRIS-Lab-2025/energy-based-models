from abc import ABC, abstractmethod
import random

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

    def __init__(self, network, energy_fn, cost_fn, config):
        super().__init__()
        self.network = network
        self.energy_fn = energy_fn
        self.cost = cost_fn
        self.config = config
        self.order = config.updater['order']
        self.iterations = config.updater['iterations']
        self._update_order()

    def _update_order(self):
        
        # determine the order in which to update the nodes
        self.update_order = []
        if self.order == 'synchronous':
            self.update_order = self.network.free_layers
        elif self.order == 'asynchronous':
            # randomize order
            layers = self.network.free_layers
            random.shuffle(layers)
            self.update_order = layers
        else:
            raise ValueError("Invalid update order")

    def step(self, S, W, B, target=None, nudging=0):
        for layer in self.update_order:
            for node in self.network.layers[layer]:
                # compute the gradient for the node
                grad = self.energy_fn.node_gradient(S, W, B, node)

                # add the cost gradient
                if layer == self.cost_fn.layer and target is not None:
                    grad += nudging * self.cost_fn.node_gradient(S, target, node)

                # pass through activation function
                grad = self.network.activation_fn[layer](grad)

                # update the state
                S[node] = grad
        return S
        

    def compute_equilibrium(self, S, W,B,target,nudging=0):
        
        # iterate for a fixed number of steps to reach equilibrium
        for i in range(self.iterations):
            S = self.step(S,W,B,target,nudging)

        return S




