from model._network import Network

class FullyConnectedNetwork(Network):
    def __init__(self, config, layer_shapes):
        number_of_neurons = 0
        for shape in layer_shapes: number_of_neurons += layer_shapes[shape].size()
        super().__init__(config, number_of_neurons)
        # TODO: Add weights between all layers
        # after initializing the weights convert to nn.Parameter

    