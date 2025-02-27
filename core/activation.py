import torch
from model._network import Network

def get_activation(network: Network):
    """Returns the given network's activation function applied to it's state"""
    return get_activation(network.activation, network.state)
    
def get_activation(activation, neurons: torch.Tensor):
    """Returns the given activation function applied to the given neuron values.

    Args:
        activation (string): the activation function to use. 'none', 'linear', 'sigmoid', 'hard_sigmoid', 'softmax', and 'dsilu' are all valid
        neurons (torch.Tensor): the state values of the desired neurons.
    """
    if activation == 'none':
        pass
    elif activation == 'linear':
        return linear(neurons)
    elif activation == 'sigmoid':
        return sigmoid(neurons)
    elif activation == 'hard_sigmoid':
        return hard_sigmoid(neurons)
    elif activation == "softmax":
        return softmax(neurons)
    elif activation == 'dsilu':
        return dSiLU(neurons)
    else: 
        raise ValueError('Unknown activation type: {}'.format(activation))




def linear(neurons: torch.Tensor):
    """Returns the value of the layer's state"""
    return neurons

def sigmoid(neurons: torch.Tensor):
    """Returns the logistic function applied to the layer's state"""
    return torch.sigmoid(4 * neurons - 2)

def hard_sigmoid(neurons: torch.Tensor):
    """Returns the value of the layer's state, clamped between 0 and 1"""
    return neurons.clamp(min=0., max=1.)

def softmax(neurons: torch.Tensor):
    """Returns the softmax function applied to the layer's state"""
    # FIXME: computing the cross-entropy loss requires computing the log first.
    # Directly computing the log_softmax would be faster and would have better numerical properties.
    return torch.nn.functional.softmax(neurons)

# dSiLULayer includes auxiliary
def auxiliary(neurons):
    torch.sigmoid(3 * neurons) * neurons

def dSiLU(neurons: torch.Tensor):
    """Returns the sigmoid-weighted linear function applied to the layer's state"""
    return auxiliary(neurons) - auxiliary(neurons - 1.)