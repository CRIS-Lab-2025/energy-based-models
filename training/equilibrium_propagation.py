import torch
from functions.energy import *
from functions.cost import *
from estimators.updater import *
from estimators.optimizer import *


class EquilibriumProp():

    def __init__(self, energy_fn: EnergyFunction, cost_fn: CostFunction, updater: Updater, optimizer: torch.optim.SGD, config):
        self.energy_fn = energy_fn
        self.cost_fn = cost_fn
        self.updater = updater
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # TODO: is this ok?

    
    def compute_gradient(self, W, S, B):
        """Estimates the parameter gradient via equilibrium propagation

        The gradient depends on the EP variant (positively-perturbed EP, negatively-perturbed EP, or centered EP) and the nudging strength.
        To get a better gradient estimate, this method must be called when the layers are at their 'free state' (equilibrium for nudging=0).
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """

        # TODO: this implementation is likely suboptimal in the case of e.g. the Readout cost function
        cost_grads = self._cost_fn._grad( W, S, B, mean=True)   # compute the direct gradient of C, if C explicitly depends on parameters
        
        # First phase: compute the first equilibrium state of the layers
        self._augmented_fn.nudging = self._first_nudging # TODO: Change the code here
        first_S = self._updater.compute_equilibrium(W, S, B)
        
        # Second phase: compute the second equilibrium state of the layers
        self._augmented_fn.nudging = self._second_nudging
        second_S = self._updater.compute_equilibrium(W, S, B)

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        param_grads = self._standard_param_grads(first_S, second_S, cost_grads, W, B)

        return param_grads + cost_grads

    def _standard_param_grads(self, first_S, second_S, cost_grads, W, B):
        """Computes the parameter gradients using the standard Equilibrium Propagation formula"""

        # Compute the energy gradients of the first state
        grads_first = self._energy_fn.full_gradient(W, first_S, B)

        # Compute the energy gradients of the second state
        grads_second = self._energy_fn.full_gradient(W, second_S, B)

        # Compute the parameter gradients
        final_grad = grads_first - grads_second / (self._second_nudging - self._first_nudging)

        return final_grad

