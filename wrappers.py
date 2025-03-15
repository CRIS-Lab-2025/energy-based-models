from model import *

class EnergyFunctionGenerator():

    def __init__(self, network) -> None:
        self.network = network
        self.network_sizes = getattr(self.network, 'layer_sizes')

    def get_network_sizes(self):
        return self.network_sizes
    
    def infer(self, x):
        init_s = torch.zeros(self.network_sizes[1:])

        return init_s
    
    def build_unclamped_energy_fn(self):

        def energy_fn(s):
            layers = list(torch.split(s, self.network_sizes, dim=1))
                
            return self.network.energy(layers)
            
        return energy_fn
    
