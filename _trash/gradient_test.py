import torch

# --- Dummy Classes to Simulate Original Code Environment ---

class DummyParameter:
    def __init__(self, tensor):
        self.tensor = tensor
    def get(self):
        return self.tensor

class DummyLayer:
    def __init__(self, state):
        self.state = state
        self.shape = state.shape

# This class holds the original functions. We assume that the overall gradient for a layer is the sum of
# _grad_layer (local bias contribution), _grad_pre, and _grad_post.
class DummyEnergyInteraction:
    def __init__(self, weight, bias, layer_pre_state, layer_post_state):
        self._weight = DummyParameter(weight)
        self._bias = DummyParameter(bias)
        self._layer_pre = DummyLayer(layer_pre_state)
        self._layer_post = DummyLayer(layer_post_state)
        # We choose the pre-synaptic layer as the one for which to broadcast the bias.
        self._layer = self._layer_pre

    def _grad_layer(self):
        # Local gradient from bias. Note: we do not modify this function.
        grad = - self._bias.get().unsqueeze(0)
        while len(grad.shape) < len(self._layer.state.shape):
            grad = grad.unsqueeze(-1)
        return grad

    def _grad_pre(self):
        # Pre-synaptic contribution: - layer_post * weight^T
        layer_post = self._layer_post.state
        dims_pre = len(self._layer_pre.shape)
        dims_post = len(self._layer_post.state.shape)
        dim_weight = len(self._weight.get().shape)
        permutation = tuple(range(dims_pre, dim_weight)) + tuple(range(dims_pre))
        return - torch.tensordot(layer_post, self._weight.get().permute(permutation), dims=dims_post)

    def _grad_post(self):
        # Post-synaptic contribution: - layer_pre * weight
        layer_pre = self._layer_pre.state
        dims_pre = len(self._layer_pre.state.shape)
        return - torch.tensordot(layer_pre, self._weight.get(), dims=dims_pre)

    def original_full_grad(self):
        """Compute the total gradient as the sum of the original parts."""
        return (self._grad_layer() + self._grad_pre() + self._grad_post()).squeeze()

# --- New Functions for Computing Gradients ---

def compute_full_gradient(weight, state, bias):
    """
    Computes the full gradient of the energy function with respect to the state.
    
    Args:
        weight (Tensor): Weight matrix of shape (n, n).
        state (Tensor): State vector of shape (n,) or state matrix of shape (batch_size, n).
        bias (Tensor): Bias vector of shape (n,).
        
    Returns:
        Tensor: Gradient of the energy function, with the same shape as `state`.
    """
    # Local bias term.
    grad = -bias
    # For a 1D state, torch.matmul treats it as a row vector.
    grad = grad - (torch.matmul(state, weight) + torch.matmul(state, weight.t()))
    return grad

def compute_node_gradient(weight, state, bias, node_index):
    """
    Computes the gradient of the energy function with respect to a single node.
    
    Args:
        weight (Tensor): Weight matrix of shape (n, n).
        state (Tensor): State vector of shape (n,) or state matrix of shape (batch_size, n).
        bias (Tensor): Bias vector of shape (n,).
        node_index (int): Index of the node.
        
    Returns:
        Tensor: Gradient for the specified node.
    """
    if state.dim() == 1:
        grad_i = -bias[node_index]
        grad_i = grad_i - torch.dot(state, weight[node_index])
        grad_i = grad_i - torch.dot(state, weight[:, node_index])
    elif state.dim() == 2:
        grad_i = -bias[node_index]
        post_contrib = (state * weight[node_index]).sum(dim=1)
        pre_contrib = (state * weight[:, node_index]).sum(dim=1)
        grad_i = grad_i - post_contrib - pre_contrib
    else:
        raise ValueError("State tensor must be 1D or 2D.")
    return grad_i

# --- Test Code ---

def main():
    # Set a random seed for reproducibility.
    torch.manual_seed(0)
    
    # Define the number of nodes.
    n = 5
    
    # Create random weight matrix, bias vector, and state vector.
    weight = torch.randn(n, n)
    bias = torch.randn(n)
    state = torch.randn(n)
    
    # Instantiate the dummy energy interaction object.
    # Here, we use the same state for both pre- and post-synaptic layers.
    interaction = DummyEnergyInteraction(weight, bias, state, state)
    
    # Compute the original full gradient (summing the bias, pre-, and post-synaptic parts).
    original_grad = interaction.original_full_grad()
    
    # Compute the full gradient using the new function.
    new_grad = compute_full_gradient(weight, state, bias)
    
    # Compute the difference.
    diff = original_grad - new_grad
    
    # Print the results.
    print("Original Gradient:\n", original_grad)
    print("New Gradient:\n", new_grad)
    print("Difference (Original - New):\n", diff)
    
    # Verify that the gradients are the same.
    if torch.allclose(original_grad, new_grad, atol=1e-6):
        print("Test Passed: The full gradients match.")
    else:
        print("Test Failed: The full gradients differ.")
    
    # Test the per-node gradient function.
    for i in range(n):
        node_grad = compute_node_gradient(weight, state, bias, i)
        # Compare with the i-th component of the full gradient.
        if torch.allclose(node_grad, new_grad[i], atol=1e-6):
            print(f"Node {i} gradient matches: {node_grad.item():.6f} vs {new_grad[i].item():.6f}")
        else:
            print(f"Node {i} gradient does NOT match: {node_grad.item():.6f} vs {new_grad[i].item():.6f}")

if __name__ == "__main__":
    main()
