import torch
from core.energy import EnergyFunction
from core.updater import Updater
from training.cost import CostFunction


class EquilibriumProp():

    def __init__(self,network, energy_fn: EnergyFunction, cost_fn: CostFunction, updater: Updater, config, optimizer):
        self.energy_fn = energy_fn
        self.cost_fn = cost_fn
        self.updater = updater
        self.network = network
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) # TODO: is this ok?
        self._variant = config.gradient_propagation['variant']
        self._nudging = config.gradient_propagation['nudging']
        self._set_nudging()

    def _set_nudging(self):
        """TODO"""
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
        _,[weight_grads_first,bias_grads_first] = self.energy_fn.full_gradient(first_S, W, B) 

        # Compute the energy gradients of the second state
        _,[weight_grads_second,bias_grads_second]= self.energy_fn.full_gradient(second_S, W, B)

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
        S_copy = S.clone()
        # First phase: compute the first equilibrium state of the layers
        first_S = self.updater.compute_equilibrium(S,W,B, target, self._first_nudging)
        
        # # Reset State before second equilibrium
        # self.network._reset_state()
        # Second phase: compute the second equilibrium state of the layers
        second_S = self.updater.compute_equilibrium(S_copy,W,B, target, self._second_nudging)

        # Compute the parameter gradients with either the standard EquilibriumProp formula, or the alternative EquilibriumProp formula
        weight_grads, bias_grads = self._standard_param_grads(first_S, second_S, W, B)

        # Apply clamping mask to the gradients
        clamped_nodes = self.network.get_clamped_indices()

        # Need to apply clamping mask for values that are fixed and non-existent in the graph
        clamped_weight_mask = (W.clone().detach()[0,::] != 0).float()
        clamped_weight_mask[clamped_nodes,:] = 0

        # If weights are zero previously, their gradients should be zero. Need to confirm if previous calculation ensures this.

        clamped_bias_mask = torch.ones_like(bias_grads)
        clamped_bias_mask[clamped_nodes] = 0

        weight_grads = weight_grads * clamped_weight_mask 
        bias_grads = bias_grads * clamped_bias_mask

        return weight_grads, bias_grads

