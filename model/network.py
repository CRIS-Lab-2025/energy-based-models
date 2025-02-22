from abc import ABC, abstractmethod
import torch

class Network(ABC):
    # TODO
    def __init__(self, config, number_of_neurons):
        # FIXME
        self.config = config
        self._state = torch.zeros(number_of_neurons, device=config.device) # TODO: do we need to set device?
        self._weights = torch.zeros((number_of_neurons, number_of_neurons))
        self._biases = torch.zeros(number_of_neurons)
        self.activations = _ # TODO

    @property
    def state(self):
        return self._state 
    
    @property
    def weights(self):
        return self._weights 
    
    @property
    def biases(self):
        return self._biases
    
    @abstractmethod
    def update(self):
        pass

    def get_energy(self):
        # TODO
        pass

    def clamp(self, neurons_to_clamp):
        # TODO
        pass