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
    def __init__(self,config):
        super().__init__()


    def energy(self, S, W, B):

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
        energy = -0.5 * torch.einsum('bi,ij,bj->b', S, W, S) - torch.einsum('bi,bi->b', S, B)
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
        bias_grad = -state.mean(dim=0)

        # For each node i, add contributions from both pre- and post-synaptic terms.
        # If state is 1D, torch.matmul treats it as a row vector.
        # Here:
        #   torch.matmul(state, weight) computes a vector with components
        #     sum_j state[j] * weight[j, i]  (i.e. the pre-synaptic contribution)
        #   torch.matmul(state, weight.t()) computes a vector with components
        #     sum_j state[j] * weight[i, j]  (i.e. the post-synaptic contribution)
        batch_size = state.shape[0]
        n_nodes = state.shape[1]
    
        # Compute the full gradient matrix directly
        # For each sample in the batch, compute the outer product of the state vector with itself
        # Then average over the batch
        weight_grad = -torch.tensordot(state, state, dims=([0], [0])) / batch_size

        return None, [weight_grad, bias_grad]


    def node_gradient(self, state, weight, bias, node_index):
        """
        Computes the gradient of the energy function with respect to a single node.

        Args:
            weight (Tensor): Weight matrix of shape (n, n) or (batch_size, n, n).
            state (Tensor): State vector of shape (n,) or state matrix of shape (batch_size, n).
            bias (Tensor): Bias vector of shape (n,) or (batch_size, n).
            node_index (int): Index of the node for which to compute the gradient.

        Returns:
            Tensor: Gradient for the given node. If `state` is batched, returns a 1D tensor
                    of gradients for each sample in the batch.
        """
        # Compute bias gradient
        bias_grad = -bias[..., node_index]  # Handles both batch and non-batch cases

        # Compute weight contribution
        if state.dim() == 1:  # Single state vector (non-batched)
            weight_grad = -(weight[node_index, :] + weight[:, node_index]) * state
        elif state.dim() == 2:  # Batched case
            pre_weight = weight[:, node_index, :]
            post_weight = weight[:, :, node_index]
            weight_grad = -(state * ( pre_weight + post_weight)).sum(dim=1)
        else:
            raise ValueError("State tensor must be 1D or 2D.")

        # Compute final gradient
        grad_i = bias_grad + weight_grad

        return grad_i, [weight_grad, bias_grad]



class ConvHopfieldEnergu(EnergyFunction):
    pass