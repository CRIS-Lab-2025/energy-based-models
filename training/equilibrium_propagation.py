import torch
from core.energy import EnergyFunction
from core.updater import Updater
from training.cost import CostFunction


import logging



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

        S1 = S[:].clone()
        W1 = W[:].clone()
        B1 = B[:].clone()
        target1 = target[:].clone()

        S2 = S[:].clone()
        W2 = W[:].clone()
        B2 = B[:].clone()
        target2 = target[:].clone()

        # print("S", S)
        # print("W", W)
        # print("B", B)
        # print("T", target)
        # print("W last", W[-1][-1])

        # print(target1)

        # First phase: compute the first equilibrium state of the layers
        first_S = self.updater.compute_equilibrium(S1,W1,B1, target1, self._first_nudging)

        # print(target2)
        
        # Second phase: compute the second equilibrium state of the layers
        second_S = self.updater.compute_equilibrium(S2,W2,B2, target2, self._second_nudging)

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

        # # Define how often to log (e.g., every 100 calls)
        # LOG_EVERY_N = 100
        # if not hasattr(self, "_gradient_log_counter"):
        #     self._gradient_log_counter = 0  # Initialize counter

        # self._gradient_log_counter += 1

        # if self._gradient_log_counter % LOG_EVERY_N == 0:
            # Compute norms of gradients for logging
        # weight_grad_norm = weight_grads.norm().item()
        # bias_grad_norm = bias_grads.norm().item()

        # # Compute norms of equilibrium states
        # first_S_norm = first_S.norm().item()
        # second_S_norm = second_S.norm().item()

        # print(
        #     # f"[Gradient Logging] Iteration: {self._gradient_log_counter}, "
        #     f"First Nudging: {self._first_nudging:.6f}, Second Nudging: {self._second_nudging:.6f}, "
        #     f"Equilibrium Norms - First: {first_S_norm:.6f}, Second: {second_S_norm:.6f}, "
        #     f"Gradient Norms - Weight: {weight_grad_norm:.6f}, Bias: {bias_grad_norm:.6f}"
        # )


        return weight_grads, bias_grads

