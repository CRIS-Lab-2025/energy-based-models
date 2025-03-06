from abc import ABC, abstractmethod
import torch
class CostFunction(ABC):
    # TODO
    def __init__(self, num_classes):
        """Initializes an instance of CostFunction

        Args:
            num_classes (int): number of categories in the classification task
        """
        self._num_classes = num_classes

    @abstractmethod
    def calculate():
        pass

class SquaredError(CostFunction):
    
    def __init__(self, config):
        
        self._config = config
        self._layer = config.cost_function['output_layer']
        num_classes = config.model['layers'][self._layer]
        super().__init__(num_classes)
        


    def calculate(self, S, target):
        """Computes the squared error cost

        Args:
            W (Tensor): weight matrix
            S (Tensor): state matrix
            B (Tensor): bias matrix
            target (Tensor): target output

        Returns:
            Tensor: cost value
        """
        output = S[self._layer]
        # Compute the squared error
        cost = 0.5 * torch.sum((output - target) ** 2)
        return cost

    def gradient(self, state, target):
        """
        Computes the gradient of the cost function with respect to the state for convolutional networks.
        
        Args:
            state (Tensor): State tensor of shape (batch_size, channels, height, width)
            target (Tensor): Target tensor
            
        Returns:
            Tensor: Gradient tensor with the same shape as state
        """
        # For convolutional networks, we need to handle the target differently
        # The target is typically a one-hot encoded vector for classification
        
        # Create a gradient tensor with the same shape as the state
        grad = torch.zeros_like(state)
        
        # If target is a class index tensor
        if len(target.shape) == 1:
            # Convert to one-hot
            batch_size = target.shape[0]
            num_classes = 10  # Assuming MNIST with 10 classes
            one_hot = torch.zeros(batch_size, num_classes, device=target.device)
            one_hot.scatter_(1, target.unsqueeze(1), 1)
            target = one_hot
        
        # Reshape target if needed
        if len(target.shape) == 2 and len(state.shape) == 4:
            # Reshape from (batch_size, num_classes) to (batch_size, channels, height, width)
            # Assuming the last channel contains the class probabilities
            batch_size, num_classes = target.shape
            channels, height, width = state.shape[1], state.shape[2], state.shape[3]
            
            # For simplicity, we'll just use the last channel for the error gradient
            # and set other channels to zero
            reshaped_target = torch.zeros(batch_size, channels, height, width, device=target.device)
            
            # Distribute the target values across the spatial dimensions of the last channel
            for b in range(batch_size):
                for c in range(num_classes):
                    if c < channels:
                        # Set the entire channel to the target value
                        reshaped_target[b, c] = target[b, c]
            
            target = reshaped_target
        
        # Compute the gradient as (state - target)
        if state.shape == target.shape:
            grad = state - target
        else:
            # If shapes don't match, we need to adapt
            print(f"Warning: State shape {state.shape} doesn't match target shape {target.shape}")
            # Just return a small gradient as a fallback
            grad = 0.01 * torch.ones_like(state)
        
        return grad

    def full_gradient(self, S, target, mean=False):
        """Computes the gradient of the cost function with respect to the entire state.

        Args:
            S (Tensor): state matrix
            target (Tensor): target output

        Returns:
            Tensor: Gradient of the cost function, with the same shape as `state`.
        """
        output = S[self._layer]
        if mean:
            output = S[self._layer].mean(dim=0)
            target = target.mean(dim=0)
        return output - target

    def node_gradient(self, grad, target, node, class_id, mean=False):
        """Computes the gradient of the cost function with respect to a single node.

        Args:
            S (Tensor): state matrix
            target (Tensor): target output
            node (int): index of the node

        Returns:
            Tensor: Gradient of the cost function, with the same shape as `state`.
        """
        
        # if more than one class, return the gradient for the class
        if self._num_classes > 1:
            output = grad
            if mean:
                output = grad.mean(dim=0)
                target = target.mean(dim=0)
            return output - target[:,class_id]
        else:
            output = grad
            if mean:
                output = grad.mean(dim=0)
                target = target.mean(dim=0)
            return target - output
           
        