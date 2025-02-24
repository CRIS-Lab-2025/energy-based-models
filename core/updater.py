from abc import ABC, abstractmethod

class Updater(ABC):
    # TODO
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _update_order(self)
        pass

    @abstractmethod
    def step(self)
        pass

    @abstractmethod
    def compute_equilibrium(self)
        pass


class FixedPointUpdater(Updater):

    def __init__(self, network, energy_fn, config):
        super().__init__()
        self.network = network
        self.energy_fn = energy_fn
        self.config = config
        self.order = config.updater['order']
        self.iterations = config.updater['iterations']
        self._update_order()

    def _update_order(self):
        self.update_order = []
        if self.order = 'synchronous':
            self.update_order = self.network.free_layers
        elif self.order = 'asynchronous':
            # randomize order
            layers = self.network.free_layers
            random.shuffle(layers)
            self.update_order = layers
        else:
            raise ValueError("Invalid update order")

    def step(self, W, S, B):
        for layer in self.update_order:
            for node in self.network.layers[layer]:
                # compute the gradient for the node
                grad = self.energy_fn.node_gradient(W, S, B, node)
                # update the node
                S[node] = grad
        return S
        

    def compute_equilibrium(self,W,S,B):
        
        for i in range(self.iterations):
            S = self.step(W,S,B)


        return S




