import os,sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model._network import Network
from model.fully_connected_network import FullyConnectedNetwork



network = FullyConnectedNetwork('', [3,4,4,2])
network.set_input(torch.Tensor([1,2,3]))
network.biases[3:11] = 0
network._weights[3,7]=2

def update(network: Network):
    network._state = torch.matmul(network.weights, network.state) + network.biases
    print(network.state)


update(network)
update(network)
update(network)
update(network)
update(network)