import torch
from dataset import Dataset
from model import Model
from core.energy import HopfieldEnergy
from core.updater import FixedPointUpdater
from training.runner import Runner
from training.cost import MeanSquaredError
from training.equilibrium_propagation import EquilibriumProp
from util.config import Config
from util.dataset import load_dataset

config = Config()
network = Model(config)
energy_fn = HopfieldEnergy(config)
cost_fn = MeanSquaredError(config)
updater = FixedPointUpdater(network, energy_fn, cost_fn, config)
differentiator = EquilibriumProp(energy_fn, cost_fn, updater, config)
W, B = network.weights, network.bias # Need to verify if pointer or new creation. It should be pointer already but sanity check. 
optimizer = torch.optim.SGD([W, B], lr=0.01)

dataset = load_dataset(config)
dataloader = Dataset(dataset, config.training.batch_size)

runner = Runner(config, network, dataloader, differentiator, updater, optimizer, inference_dataloader=None)
runner.run_training()