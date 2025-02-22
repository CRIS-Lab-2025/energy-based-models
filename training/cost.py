from abc import ABC, abstractmethod

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
    # TODO
    def calculate():
        pass