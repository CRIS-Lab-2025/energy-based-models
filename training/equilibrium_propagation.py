import torch
from functions.energy import *
from functions.cost import *
from estimators.updater import *
from estimators.optimizer import *


class EquilibriumProp():

    def __init__(self, energy_fn: EnergyFunction, cost_fn: CostFunction, updater: Updater, config):
        self.energy_fn = energy_fn
        self.cost_fn = cost_fn
        self.updater = updater
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # TODO: is this ok?
        self._variant = config.propagation['variant']
        self._nudging = config.propagation['nudging']

    def _set_nudging(self):


        if self._variant == "positive":
            self._first_nudging = 0.
            self._second_nudging = self._nudging
        elif self._variant == "negative":
            self._first_nudging = -self._nudging
            self._second_nudging = 0.
        elif self._variant == "centered":
            self._first_nudging = -self._nudging
            self._second_nudging = self._nudging
        else:
            # TODO: this raise of error is redundant
            raise ValueError("expected 'positive', 'negative' or 'centered' but got {}".format(self._variant))

    def _standard_param_grads(self, first_S, second_S, W, B):
        """Computes the parameter gradients using the standard Equilibrium Propagation formula"""

        # Compute the energy gradients of the first state
        _,[weight_grads_first,bias_grads_first] = self._energy_fn.full_gradient(W, first_S, B) 

        # Compute the energy gradients of the second state
        _,[weight_grads_second,bias_grads_second]= self._energy_fn.full_gradient(W, second_S, B)

        # Compute the parameter gradients
        weight_grads = weight_grads_second - weight_grads_first / (self._second_nudging - self._first_nudging)
        bias_grads = bias_grads_second - bias_grads_first / (self._second_nudging - self._first_nudging)

        return weight_grads, bias_grads
    

    def compute_gradient(self, S, W, B, target=None):
        """Estimates the parameter gradient via equilibrium propagation

        The gradient depends on the EP variant (positively-perturbed EP, negatively-perturbed EP, or centered EP) and the nudging strength.
        To get a better gradient estimate, this method must be called when the layers are at their 'free state' (equilibrium for nudging=0).
        
        Returns:
            param_grads: list of Tensor of shape param_shape and type float32. The parameter gradients
        """
        # First phase: compute the first equilibrium state of the layers
        first_S = self._updater.compute_equilibrium(W, S, B, target, self._first_nudging)
        
        # Second phase: compute the second equilibrium state of the layers
        second_S = self._updater.compute_equilibrium(W, S, B, target, self._second_nudging)

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        weight_grads, bias_grads = self._standard_param_grads(first_S, second_S, cost_grads, W, B)

        # Doesn't calculate grads for biases I think need to fix that. There is something iffy about the bias gradients calculation part. 
        clamped_nodes = self.network.clamped_weights()

        # Need to apply clamping mask for values that are fixed.
        clamped_weight_mask = torch.ones_like(W)
        clamped_weight_mask[clamped_nodes] = 0
        # If weights are zero previously, their gradients should be zero. Need to confirm if previous calculation ensures this.

        clamped_bias_mask = torch.ones_like(B)
        clamped_bias_mask[clamped_nodes] = 0

        weight_grads = weight_grads * clamped_mask 
        bias_grads = bias_grads * clamped_bias_mask

        return weight_grads, bias_grads

