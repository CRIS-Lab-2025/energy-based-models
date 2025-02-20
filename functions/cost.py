from abc import ABC, abstractmethod

class CostFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate():
        pass

class SquaredError(CostFunction):
    def calculate():
        pass