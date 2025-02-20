import torch

def get_optimizer(model, cost_function):
    """Returns an optimizer for the model and cost function based on the configuration.

    The model configuration should have:
      - optimizer_type: either 'sgd' or 'adam'
      - optimizer: a dict with keys:
          - 'learning_rates_weights' (list of floats)
          - 'learning_rates_biases' (list of floats)
          - 'momentum' (float)
          - 'weight_decay' (float)
          
    Args:
        model (Model): the model whose parameters are optimized.
        cost_function (CostFunction): the cost function whose parameters are optimized.

    Returns:
        Optimizer: the configured optimizer.
    """
    config = model.config
    if config.optimizer_type == 'sgd':
        return SGDOptimizer(model, cost_function)
    elif config.optimizer_type == 'adam':
        return AdamOptimizer(model, cost_function)
    else:
        raise ValueError("Unknown optimizer: {}".format(config.optimizer_type))


class BaseOptimizer:
    """
    Base class for optimizers that abstracts common configuration and parameter extraction.

    This class:
      - Extracts the learning rates, momentum, and weight decay from model.config.optimizer.
      - Combines parameters from the energy function and cost function.
      - Initializes the underlying torch optimizer with per-parameter learning rates.
      
    Args:
        model (Model): the model, which contains the configuration.
        cost_function (CostFunction): the cost function.
        torch_optimizer_class (type): a torch.optim optimizer class (e.g. torch.optim.SGD).
        energy_fn_getter (callable): a function that retrieves the energy function from the model.
    """
    def __init__(self, model, cost_function, torch_optimizer_class, energy_fn_getter):
        self.config = model.config
        optimizer_config = self.config.optimizer
        
        # Combine the learning rates for weights and biases.
        self.learning_rates = optimizer_config['learning_rates_weights'] + optimizer_config['learning_rates_biases']
        self.momentum = optimizer_config["momentum"]
        self.weight_decay = optimizer_config["weight_decay"]
        
        # Get the energy function from the model.
        self.energy_fn = energy_fn_getter(model)
        
        # Gather parameters from both the energy function and the cost function.
        params = self.energy_fn.params() + cost_function.params()
        params = [{"params": param.state, "lr": lr} for param, lr in zip(params, self.learning_rates)]
        
        # Initialize the underlying torch optimizer.
        torch_optimizer_class.__init__(self, params, lr=0.1,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        
        # Store for printing.
        self._learning_rates = self.learning_rates
        self._momentum = self.momentum
        self._weight_decay = self.weight_decay

    def __str__(self):
        return "{} -- initial learning rates = {}, momentum = {}, weight_decay = {}".format(
            self.__class__.__name__,
            self._learning_rates,
            self._momentum,
            self._weight_decay
        )


class SGDOptimizer(BaseOptimizer, torch.optim.SGD):
    def __init__(self, model, cost_function):
        # For SGD, assume the energy function is accessible as model.function.
        super().__init__(model, cost_function, torch.optim.SGD, lambda m: m.function)


class AdamOptimizer(BaseOptimizer, torch.optim.Adam):
    def __init__(self, model, cost_function):
        # For Adam, assume the energy function is accessible as model._function().
        super().__init__(model, cost_function, torch.optim.Adam, lambda m: m._function())
