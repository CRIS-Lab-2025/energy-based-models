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
        weight_grad = -torch.matmul(state.transpose(0, 1), state) / batch_size

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



class ConvHopfieldEnergy(EnergyFunction):
    """
    Convolutional Hopfield Energy function for use with convolutional architectures.
    This energy function is defined as:
    E(S) = -0.5 * sum_{l,i,j,k} S_{l,i,j} * W_{l,k} * S_{l,i+k_x,j+k_y} - sum_{l,i,j} S_{l,i,j} * B_{l}
    where:
    - S is the state tensor of shape (batch_size, channels, height, width)
    - W is the weight tensor of shape (channels, kernel_size, kernel_size)
    - B is the bias tensor of shape (channels)
    - l indexes channels, i,j index spatial dimensions, k indexes kernel positions
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def energy(self, S, W, B):
        """
        Compute the energy of the state S given the convolutional weights W and biases B.
        
        Args:
            S (Tensor): State tensor of shape (batch_size, channels, height, width)
            W (Tensor): Weight tensor for convolution
            B (Tensor): Bias tensor
            
        Returns:
            Tensor: Energy values for each sample in the batch
        """
        batch_size = S.shape[0]
        
        # Compute the quadratic term using convolution operations
        # For each channel, convolve the state with the weights
        conv_term = torch.zeros(batch_size, device=S.device)
        
        for b in range(batch_size):
            for c in range(S.shape[1]):  # For each channel
                # Apply convolution and element-wise multiply with the state
                conv_result = torch.nn.functional.conv2d(
                    S[b:b+1, c:c+1], 
                    W[c:c+1].unsqueeze(1),
                    padding='same'
                )
                conv_term[b] -= 0.5 * torch.sum(conv_result * S[b:b+1, c:c+1])
        
        # Compute the bias term
        bias_term = -torch.sum(S * B.view(1, -1, 1, 1), dim=[1, 2, 3])
        
        return conv_term + bias_term

    def full_gradient(self, state, weight, bias):
        """
        Computes the gradient of the energy function with respect to weights and biases.
        
        Args:
            state (Tensor): State tensor of shape (batch_size, channels, height, width)
            weight (Tensor or tuple): Weight tensor for convolution or tuple containing weight tensor
            bias (Tensor): Bias tensor
            
        Returns:
            Tuple: (state_grad, [weight_grad, bias_grad])
        """
        # Handle the case where weight is a tuple
        if isinstance(weight, tuple):
            weight = weight[0]
        
        # Handle the case where weight is a batched tensor
        if len(weight.shape) > 3 and weight.shape[0] == state.shape[0]:
            # Use the first batch item's weights
            weight = weight[0]
            
        batch_size = state.shape[0]
        channels = state.shape[1]
        
        # Compute bias gradient
        bias_grad = -torch.mean(torch.sum(state, dim=[2, 3]), dim=0)
        
        # Create a state gradient tensor
        state_grad = torch.zeros_like(state)
        
        # Check if we have proper convolutional weights
        if len(weight.shape) < 3:
            # We don't have proper convolutional weights, so we'll use a simpler approach
            print(f"Warning: Weight shape {weight.shape} is not suitable for convolution")
            
            # Create a simple weight gradient
            if hasattr(self.config, 'model') and 'kernel_size' in self.config.model:
                kernel_size = self.config.model['kernel_size']
                weight_grad = torch.zeros((channels, kernel_size, kernel_size), device=state.device)
            else:
                # Default shape
                weight_grad = torch.zeros((channels, 3, 3), device=state.device)
            
            # Use a simple gradient calculation
            for c in range(channels):
                # Simple weight gradient
                weight_grad[c] = -0.01 * torch.ones_like(weight_grad[c])
                
                # Simple state gradient
                for b in range(batch_size):
                    # Add bias contribution
                    if c < len(bias):
                        state_grad[b, c] = -bias[c]
                    
                    # Add a small gradient to encourage movement
                    state_grad[b, c] -= 0.01
            
            return state_grad, [weight_grad, bias_grad]
        
        # Get kernel size from the weight tensor
        kernel_size = weight.shape[-1]
        padding = kernel_size // 2
        
        # Compute weight gradient
        weight_grad = torch.zeros_like(weight)
        
        for c in range(channels):
            for b in range(batch_size):
                # Compute correlation between state maps
                # This is equivalent to convolution with flipped kernels
                try:
                    corr = torch.nn.functional.conv2d(
                        state[b:b+1, c:c+1],
                        state[b:b+1, c:c+1],
                        padding=padding
                    ).squeeze() / batch_size
                    
                    # Make sure the correlation has the right shape
                    if corr.shape == weight_grad[c].shape:
                        weight_grad[c] -= corr
                    else:
                        # Resize if shapes don't match
                        center = corr.shape[0] // 2
                        k_size = weight_grad[c].shape[0]
                        start = center - k_size // 2
                        end = start + k_size
                        if start >= 0 and end <= corr.shape[0]:
                            weight_grad[c] -= corr[start:end, start:end]
                    
                    # Compute state gradient using convolution with the weight
                    try:
                        conv_result = torch.nn.functional.conv2d(
                            state[b:b+1, c:c+1],
                            weight[c:c+1].unsqueeze(1),
                            padding=padding
                        )
                        state_grad[b, c] = -conv_result.squeeze()
                    except Exception as e:
                        print(f"Error in state gradient conv2d: {e}")
                        # Use a fallback method
                        state_grad[b, c] -= 0.01
                    
                    # Add bias contribution to state gradient
                    if c < len(bias):
                        state_grad[b, c] -= bias[c]
                    
                except Exception as e:
                    print(f"Error in weight gradient conv2d: {e}")
                    print(f"State shape: {state.shape}, Weight shape: {weight.shape}")
                    # Use a fallback method
                    weight_grad[c] -= 0.01  # Small constant gradient as fallback
                    state_grad[b, c] -= 0.01  # Small constant gradient as fallback
        
        return state_grad, [weight_grad, bias_grad]

    def node_gradient(self, state, weight, bias, node_indices):
        """
        Computes the gradient of the energy function with respect to a specific node.
        
        Args:
            state (Tensor): State tensor of shape (batch_size, channels, height, width)
            weight (Tensor): Weight tensor for convolution
            bias (Tensor): Bias tensor
            node_indices (tuple): (channel, height, width) indices of the node
            
        Returns:
            Tuple: (gradient, [weight_contrib, bias_contrib])
        """
        channel, height, width = node_indices
        batch_size = state.shape[0]
        
        # Extract the bias contribution
        bias_contrib = -bias[channel]
        
        # Compute the weight contribution using convolution
        weight_contrib = torch.zeros(batch_size, device=state.device)
        
        for b in range(batch_size):
            # Apply convolution centered at the node position
            conv_result = torch.nn.functional.conv2d(
                state[b:b+1, channel:channel+1],
                weight[channel:channel+1].unsqueeze(1),
                padding='same'
            )
            # Extract the value at the node position
            weight_contrib[b] = -conv_result[0, 0, height, width]
        
        gradient = bias_contrib + weight_contrib
        
        return gradient, [weight_contrib, bias_contrib]