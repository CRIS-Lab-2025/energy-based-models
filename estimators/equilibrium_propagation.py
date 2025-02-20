import torch
from functions.energy import *
from functions.cost import *
from estimators.updater import *
from estimators.optimizer import *

class EquilibriumProp():
    # TODO
    def __init__(self, energy_fn: EnergyFunction, cost_fn: CostFunction, updater: Updater, optimizer: torch.optim.SGD):
        self.energy_fn = energy_fn
        self.cost_fn = cost_fn
        self.updater = updater
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # TODO: is this ok?


class HopfieldEqProp(EquilibriumProp):
    # TODO
    def __init__(self, cost_fn, updater, optimizer):
        energy_fn = HopfieldEnergy()
        super().__init__(energy_fn, cost_fn, updater, optimizer)