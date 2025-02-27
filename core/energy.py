from abc import ABC, abstractmethod
import torch

class EnergyFunction(ABC):
    # TODO
    def __init__(self):
        pass
    
    @abstractmethod
    def energy():
        pass

    @abstractmethod
    def full_gradient():
        pass

    def node_gradient():
        pass

class HopfieldEnergy(EnergyFunction):
    # TODO
    def __init__(self):
        super().__init__()


    def energy(self, W, S):

        # Compute the energy of the state S given the weight matrix W.
        # The energy is defined as:
        #   E(S) = -0.5 * S^T W S - S^T b
        # where:
        #   - S is the state vector
        #   - W is the weight matrix
        #   - b is the bias vector
        #   - ^T denotes the transpose operation
        #   - * denotes matrix multiplication
        #   - -0.5 is a scaling factor to avoid double-counting connections
        # The energy is a scalar value for each state in the batch.
        energy = -0.5 * torch.einsum('bi,ij,bj->b', S, W, S) - torch.einsum('bi,bi->b', S, bias)
        return energy

    def full_gradient(self, state, weight, bias):
        """
        Computes the gradient of the energy function with respect to the entire state.
        
        Args:
            weight (Tensor): Weight matrix of shape (n, n).
            state (Tensor): State vector of shape (n,) or state matrix of shape (batch_size, n).
            bias (Tensor): Bias vector of shape (n,).
            
        Returns:
            Tensor: Gradient of the energy function, with the same shape as `state`.
        """
        # Compute the bias term. If state is batched, bias will be broadcasted.
        bias_grad = -bias

        # For each node i, add contributions from both pre- and post-synaptic terms.
        # If state is 1D, torch.matmul treats it as a row vector.
        # Here:
        #   torch.matmul(state, weight) computes a vector with components
        #     sum_j state[j] * weight[j, i]  (i.e. the pre-synaptic contribution)
        #   torch.matmul(state, weight.t()) computes a vector with components
        #     sum_j state[j] * weight[i, j]  (i.e. the post-synaptic contribution)
        weight_grad = - (torch.matmul(state, weight) + torch.matmul(state, weight.t()))
        grad = bias_grad + weight_grad
        return grad, [weight_grad, bias_grad]


    def node_gradient(self, state, weight, bias, node_index):
        """
        Computes the gradient of the energy function with respect to a single node.
        
        Args:
            weight (Tensor): Weight matrix of shape (n, n).
            state (Tensor): State vector of shape (n,) or state matrix of shape (batch_size, n).
            bias (Tensor): Bias vector of shape (n,).
            node_index (int): Index of the node for which to compute the gradient.
            
        Returns:
            Tensor: Gradient for the given node. If `state` is batched, returns a 1D tensor
                    of gradients for each sample in the batch.
        """
        # For node i, the gradient is:
        #   - bias[i] - (sum_j weight[i,j] * state[j] + sum_j weight[j,i] * state[j])
        bias_grad = -bias[node_index]

        if state.dim() == 1:
            weight_grad = -(weight[node_index] + weight[:, node_index])
            grad_i = bias_grad + weight_grad
        elif state.dim() == 2:
            weight_grad = -(weight[node_index] + weight[:, node_index])
            grad_i = bias_grad + (state * weight_grad).sum(dim=1)
        else:
            raise ValueError("State tensor must be 1D or 2D.")
        
        return grad_i, [weight_grad, bias_grad]


class ConvHopfieldEnergu(EnergyFunction):
    pass