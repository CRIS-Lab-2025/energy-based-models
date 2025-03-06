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

        return target - grad
        # if self._num_classes > 1:
        #     output = grad
        #     # if mean:
        #     #     output = grad.mean(dim=0)
        #     #     target = target.mean(dim=0)
        #     return output - target[:,class_id]
        # else:
        #     output = grad
        #     # if mean:
        #     #     output = grad.mean(dim=0)
        #     #     target = target.mean(dim=0)
        #     return target - output
           
        