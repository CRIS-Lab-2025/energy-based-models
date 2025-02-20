import torch
from model.layer import Layer, DropOutLayer

def get_activation(layer: Layer):
    """Returns the given layer's activation function applied to it's state"""
    if layer.activation == 'none':
        pass
    elif layer.activation == 'linear':
        return linear(layer)
    elif layer.activation == 'sigmoid':
        return sigmoid(layer)
    elif layer.activation == 'hard_sigmoid':
        return hard_sigmoid(layer)
    elif layer.activation == "softmax":
        return softmax(layer)
    elif layer.activation == 'dsilu':
        return dSiLU(layer)
    elif layer.activation == 'dropout':
        return dropout(layer)
    elif layer.activation == 'nonlinear_restitive':
        return nonlinear_restitive(layer)
    else: 
        raise ValueError('Unknown activation type: {}'.format(layer.activation_type))



def linear(layer: Layer):
    """Returns the value of the layer's state"""
    return layer._state

def sigmoid(layer: Layer):
    """Returns the logistic function applied to the layer's state"""
    return torch.sigmoid(4 * layer._state - 2)

def hard_sigmoid(layer: Layer):
    """Returns the value of the layer's state, clamped between 0 and 1"""
    return layer._state.clamp(min=0., max=1.)

def softmax(layer: Layer):
    """Returns the softmax function applied to the layer's state"""
    # FIXME: computing the cross-entropy loss requires computing the log first.
    # Directly computing the log_softmax would be faster and would have better numerical properties.
    return torch.nn.functional.softmax(layer._state)

# dSiLULayer includes auxiliary
def auxiliary(state):
    torch.sigmoid(3 * state) * state

def dSiLU(layer: Layer):
    """Returns the sigmoid-weighted linear function applied to the layer's state"""
    return auxiliary(layer._state) - auxiliary(layer._state - 1.)

# dropout includes draw_mask
def dropout(layer: DropOutLayer):
    """Activate the dropout layer"""
    return 1./(1.-layer._sparsity) * layer._mask * layer._layer.activate()


def nonlinear_restitive(layer: Layer):
    """Returns the value of the layer's state, clamped between 0 and +infinity for excitatory units, and clamped between -infinity and 0 for inhibitory units"""
        
    dimension = layer._shape[0] // 2  # number of excitatory units = number of inhibitory units = number of units / 2
    excitatory = layer._state[:,:dimension].clamp(min=0., max=None)  # the first half of the units are excitatory units
    inhibitory = layer._state[:,dimension:].clamp(min=None, max=0.)  # the second half of the units are inhibitory units
    return torch.cat((excitatory, inhibitory), 1)  # we concatenate excitatory and inhibitory units