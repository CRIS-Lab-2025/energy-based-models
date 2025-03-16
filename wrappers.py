from model import *

class EnergyFunctionGenerator():

    def __init__(self, network) -> None:
        self.network = network
        self.network_sizes = self.network.layer_sizes

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

    def input_clamped_energy_fn(self, x):

        def energy_fn(s):
            h = torch.cat((x, s), dim=1)
            layers = list(torch.split(h, self.network_sizes, dim=1))

            return self.network.energy(layers)
        
        return energy_fn
    







