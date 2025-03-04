import torch
import os,sys 
import torch.nn as nn
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model._network import Network

class FullyConnectedNetwork(Network):
    """TODO: Class description"""
    def __init__(self, config, layer_shapes=None, activation='hard-sigmoid', pool_type='conv_max_pool', weight_gains=[0.6, 0.6, 1.5]):
        if layer_shapes is None:
            layer_shapes = config.model['layers']
        num_neurons = sum(layer_shapes)

        self._layer_shapes = layer_shapes
        self.input_shape = layer_shapes[0]
        self.num_layers = len(layer_shapes)
        self.layers = [range(self._layer_start(l),self._layer_end(l)) for l in range(self.num_layers)]

        self.batch_size = config.training['batch_size']
        super().__init__(config, num_neurons,batch_size=self.batch_size)
        self._init_edges()
        self.free_layers = list(range(1,self.num_layers))
    
    def _init_edges(self):
        """Create an edge from each node of each layer to each node of the subsequent layer.  
        """
        for input_index in range(self._layer_shapes[0]): 
            # ensure inputs don't change
            with torch.no_grad():
                self._weights[input_index,input_index]=1

        weight_gains = np.sqrt([2.0 / (self._layer_shapes[i]) for i in range(self.num_layers-1)])
        for layer in range(self.num_layers-1):
            # create edges from each layer to the next
            col_start = self._layer_start(layer)
            col_end = self._layer_end(layer)
            row_start = self._layer_start(layer+1)
            row_end = self._layer_end(layer+1)
            # do kaiming initialization
            nn.init.kaiming_normal_(self._weights[row_start:row_end, col_start:col_end], a=weight_gains[layer])

            with torch.no_grad():
                self._weights[row_start:row_end, col_start:col_end] = torch.clamp(self._weights[row_start:row_end, col_start:col_end], max=0.32, min=-0.32)
        
    
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
        if input.shape[1] is not self.input_shape: 
            raise ValueError('Wrong number of inputs. Got {} but expected {}'.format(input.shape[0], self.input_shape))
        self.state[:,:self.input_shape] = input
        return self.state
    
    def clamp_layer(self, l):
        """Clamp all neurons in the given layer"""
        self.clamp(self.get_layer_indices(l))