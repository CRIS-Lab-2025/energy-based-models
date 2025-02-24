from abc import ABC, abstractmethod

class CostFunction(ABC):
    # TODO
    def __init__(self, num_classes):
        """Initializes an instance of CostFunction

        Args:
            num_classes (int): number of categories in the classification task
        """
        self._num_classes = num_classes
        self._layer = None

    @abstractmethod
    def calculate():
        pass

class SquaredError(CostFunction):
    
    def __init__(self, network, config):
        super().__init__()
        self._network = network
        self._config = config
        self._layer = network.layers[-1]

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

    def node_gradient(self, S, target, node, mean=False):
        """Computes the gradient of the cost function with respect to a single node.

        Args:
            S (Tensor): state matrix
            target (Tensor): target output
            node (int): index of the node

        Returns:
            Tensor: Gradient of the cost function, with the same shape as `state`.
        """
        output = S[self._layer][node]
        if mean:
            output = S[self._layer][:, node].mean()
            target = target.mean()
        return output - target[node]
        