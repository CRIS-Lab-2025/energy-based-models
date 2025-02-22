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
    
    def create_edge(self, pre_index, post_index, weight=1):
        """Create an edge initialized with the specified weight value.

        Args:
            pre_index (int): the index of the source neuron
            post_index (int): the index of the recieving neuron
            weight (int, optional): the initial value of this weight. Default=1
        """
        self._weights[pre_index][post_index] = weight
    
    @abstractmethod
    def update(self):
        pass

    def clamp(self, neurons_to_clamp):
        # TODO
        pass