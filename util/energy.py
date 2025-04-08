import torch
from activation import *

def hopfield(layers, weights, biases):
    """Compute the energy function E for the current layers."""
    # squared_norm: for each layer, sum of squares of each row, then sum over layers.
    squared_norm = sum([(layer * layer).sum(dim=1) for layer in layers]) / 2.0
    # linear_terms: for each layer, compute dot(layer, bias).
    linear_terms = -sum([torch.matmul(layer, b) for layer, b in zip(layers, biases)])
    # quadratic_terms: for each adjacent pair of layers.
    quadratic_terms = -sum([
        ((torch.matmul(pre, W)) * post).sum(dim=1)
        for pre, W, post in zip(layers[:-1], weights, layers[1:])
    ])
    return squared_norm + linear_terms + quadratic_terms

def bengio(layers, weights, biases, act):

    squared_norm = sum([(layer * layer).sum(dim=1) for layer in layers]) / 2.0
    linear_terms = -sum([torch.matmul(get_activation(act, layer), b) for layer, b in zip(layers, biases)])
    quadratic_terms = -sum([
        ((torch.matmul(get_activation(act, pre), W)) * get_activation(act, post)).sum(dim=1)
        for pre, W, post in zip(layers[:-1], weights, layers[1:])
    ])
    return squared_norm + linear_terms + quadratic_terms


def jaynes():
    #TODO
    pass