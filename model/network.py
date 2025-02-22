from abc import ABC, abstractmethod
import torch

class Network(ABC):
    # TODO
    def __init__(self, config, weights, number_of_neurons):
        self.config = config
        self.state = torch.zeros(number_of_neurons)
