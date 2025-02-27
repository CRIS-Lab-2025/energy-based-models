import torch
import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model._network import Network

class FullyConnectedNetwork(Network):
    def __init__(self, config, layer_shapes, activation='hard-sigmoid', pool_type='conv_max_pool', weight_gains=[0.6, 0.6, 1.5]):
        num_neurons = sum(layer_shapes)
        self._layer_shapes = layer_shapes
        self.input_shape = layer_shapes[0]
        self.num_layers = len(layer_shapes)
        
        super().__init__(config, num_neurons, activation=activation)
        self._init_edges()

        # TODO: after initializing the weights convert to nn.Parameter
    
    def _init_edges(self):
        """Create an edge from each node of each layer to each node of the subsequent layer.  
        """
        for input_index in range(self._layer_shapes[0]): 
            # ensure inputs don't change
            self._weights[input_index,input_index]=1

        for layer in range(self.num_layers-1):
            # create edges from each layer to the next
            col_start = self._layer_start(layer)
            col_end = self._layer_end(layer)
            row_start = self._layer_start(layer+1)
            row_end = self._layer_end(layer+1)
            self._weights[row_start:row_end, col_start:col_end] = 1
        
    
    def _layer_start(self, l):
        """Returns the index of the first element in the specified layer.
        
        Args:
            l: (int): the desired layer
        """
        if l == 0: return 0
        return sum(self._layer_shapes[0:l])
    
    def _layer_end(self, l):
        """Returns the index of the last element in the specified layer.
        
        Args:
            l: (int): the desired layer
        """
        return self._layer_shapes[l] + self._layer_start(l)
    
    def get_layer_state(self, l):
        """Returns the current state of the given layer. Layer 0 is the input layer.

        Args:
            l (int): the desired layer
        """
        return self._state[self._layer_start(l):self._layer_end(l)]
    
    def get_layer_indices(self, l):
        return range(self._layer_start(l),self._layer_end(l))
    
    def set_input(self, input: torch.Tensor):
        """Update the network state by setting the input to the values 
        in the given tensor.
        """
        if input.shape[0] is not self.input_shape: 
            raise ValueError('Wrong number of inputs. Got {} but expected {}'.format(input.shape[0], self.input_shape))
        self.state[:self.input_shape] = input
    
    def clamp_layer(self, l):
        """Clamp all neurons in the given layer"""
        self.clamp(self.get_layer_indices(l))
