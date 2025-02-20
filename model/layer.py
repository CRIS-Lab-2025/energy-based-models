from abc import ABC, abstractmethod
from model.variable import *
from functions.activation import *

class Layer(Variable, ABC):
    """
    Class used to implement a layer of units

    Attributes
    ----------
    _counter (int): the number of Layers instanciated so far
    name (str): the layer's name (used e.g. to identify the layer in tensorboard)

    Methods
    -------
    activate():
        Applies a nonlinearity to the layer's state
    """

    _counter = 0  # the number of instantiated Layers

    def __init__(self, shape, batch_size=1, device=None, activation='hard-sigmoid'):
        """Initializes an instance of Layer

        Args:
            shape (tuple of int): shape of the tensor used to represent the state of the layer
            batch_size (int, optional): the size of the current batch processed. Default: 1
            device (str, optional): the device on which to run the layer's tensor. Either `cuda' or `cpu'. Default: None
            activation (str, optional): the layer's activation function. Default: 'hard-sigmoid'.
        """
        Variable.__init__(self, shape)
        self.activation = activation
        self._name = 'Layer_{}'.format(Layer._counter)  # the name of the layer is numbered for ease of identification

        self.init_state(batch_size, device)
        
        Layer._counter += 1  # the number of instanciated layers is now increased by 1

    @property
    def name(self):
        """Get the name of the layer"""
        return self._name

    def init_state(self, batch_size, device):
        """Initializes the state of the layer to zero

        Args:
            batch_size (int): size of the mini-batch of examples
            device (str): Either 'cpu' or 'cuda'
        """

        shape = (batch_size,) + self._shape
        self._state = torch.zeros(shape, requires_grad=False, device=device)
    
    def activate(self):
        """Returns the activation function applied to the layer's state"""
        return get_activation(self)
    
class DropOutLayer(Layer):
    """
    Class used to implement a layer with dropout

    Attributes
    ----------
    _layer (Layer): the layer on which we apply dropout
    _sparsity (float, optional): the level of sparsity of the layer. Default: 0.5

    Methods
    -------
    draw_mask():
        Draw a new mask for dropout
    """

    def __init__(self, layer: Layer, sparsity=0.5):
        """Initializes an instance of Layer

        Args:
            layer (Layer): the layer on which we want to apply dropout
            sparsity (float, optional): the level of sparsity of the layer. Default: 0.5
        """
        self._layer = layer
        self.activation = 'dropout'
        self._sparsity = sparsity
 
    def draw_mask(self):
        """Draw a new mask for dropout"""
        self._mask = (torch.rand(*self._layer.state._shape) > self._sparsity).float()
