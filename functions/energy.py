from abc import ABC, abstractmethod

class EnergyFunction(ABC):
    # TODO
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate():
        pass

class HopfieldEnergy(EnergyFunction):
    # TODO
    def __init__(self):
        super().__init__()
