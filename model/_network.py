from abc import ABC, abstractmethod
import torch
from util.config import Config

class Network(ABC):
    """
    Class used to #TODO
    """
    def __init__(self, config: Config, num_neurons, batch_size):
        self.config = config
        self._state = torch.zeros(batch_size, num_neurons, device=config.device)
        self._weights = torch.nn.Parameter(torch.zeros((num_neurons, num_neurons), device=config.device))
        self._biases = torch.nn.Parameter(torch.zeros(num_neurons, device=config.device))
        self.activation = config.model['activation']

    @property
    def state(self):
        return self._state 
    
    @property
    def weights(self):
        return self._weights 
    
    @property
    def biases(self):
        return self._biases
    
    def set_input(self, input: torch.Tensor):
        """Update the network state by setting the input to the values 
        in the given tensor.
        """
        self.state[:input.shape[0]] = input
    
    def create_edge(self, pre_index, post_index, weight=1):
        """Create an edge initialized with the specified weight value.

        Args:
            pre_index (int): the index of the source neuron
            post_index (int): the index of the recieving neuron
            weight (int, optional): the initial value of this weight. Default=1
        """
        self._weights[post_index][pre_index] = weight

    def clamp(self, neurons):
        """Clamp the given neurons so that no matter what their value does not change.

        Args:
            neurons (List[int]): a list of the indices of the neurons that should be clamped
        """
        for neuron in neurons: 
            self._weights[neuron] = torch.zeros_like(self._weights[neuron]) 
            self._weights[neuron][neuron] = 1
        # TODO: If we want to clamp neurons DURING training we need to make sure that 1 value never updates
        # TODO: To implement the above, we may as well come up with a way to clamp WEIGHTS during training

    def get_clamped_indices(self):
        """Return a list of indices of every clamped neuron in the network."""
        clamped = []
        for i in range(self.state.shape[1]):
            if torch.sum(torch.abs(self.weights[i])) == 1 and self.weights[i][i] == 1:
                clamped.append(i)
        return clamped
