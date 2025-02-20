from abc import ABC, abstractmethod

class Updater(ABC):
    # TODO
    def __init__(self):
        super().__init__()

    @abstractmethod
    def pre_activate():
        pass

    def step():
        '''
        Perform a step. Update weights and biases accordingly

        TODO:
        1. figure out some ordering of the layers (fwd/bkwd/sync/async)
        2. For each layer (in order): Do pre-activation, then activation
        '''
        
        pass
