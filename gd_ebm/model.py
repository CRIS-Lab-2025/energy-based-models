import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyBasedModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, optimizer, beta=0.1, dt=0.1, n_steps=20):
        """
        Initialize an Energy-Based Model with equilibrium propagation.
        
        Args:
            input_size (int): Size of input layer
            hidden_sizes (list): List of hidden layer sizes
            optimizer: PyTorch optimizer (will be set separately)
            beta (float): Influence of the output cost on the energy
            dt (float): Time step for dynamics
            n_steps (int): Number of steps for reaching equilibrium
        """
        super().__init__()
        import torch._dynamo
        self = torch.compile(self, mode="reduce-overhead")

        self.layer_sizes = [input_size] + hidden_sizes
        #print(self.layer_sizes)
        self.beta = beta
        self.dt = dt
        self.n_steps = n_steps
        self.step_size = dt 
        self.training_mode = False
        self.debug = False
        # Initialize weights and biases for symmetric connections
        # We use separate weights for forward and backward to maintain symmetry explicitly
        self.weights = nn.ParameterList([
            nn.Parameter(
            torch.empty(self.layer_sizes[i], self.layer_sizes[i+1]).uniform_(-1 / torch.sqrt(torch.tensor(self.layer_sizes[i], dtype=torch.float32)), 
                                             1 / torch.sqrt(torch.tensor(self.layer_sizes[i], dtype=torch.float32)))
            ) for i in range(len(self.layer_sizes)-1)
        ])
        
        # Initialize biases for each layer
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(size))
            for size in self.layer_sizes[1:]
        ])
        
        # States will be initialized during forward pass based on batch size
        self.states = None
        
        # Initialize optimizer
        self.optimizer = optimizer
    
    def gradient(self, inputs):
        first_states, second_states = inputs
        first_states, second_states = [self.activation(s) for s in first_states], [self.activation(s) for s in second_states]

        
        weights_gradients = []
        biases_gradients = []
        
        for i in range(len(self.weights)):
            # Compute batch-wise outer products and take mean over batch dimension

            # outer products
            weight_grad = torch.einsum('bi,bj->ij', second_states[i], second_states[i+1])
            weight_grad -= torch.einsum('bi,bj->ij', first_states[i],  first_states[i+1])
            weight_grad.div_(self.beta * first_states[0].shape[0])           # mean & beta in‑place

            weights_gradients.append(weight_grad)
            
            bias_grad = (second_states[i+1] - first_states[i+1]).mean(dim=0)
            biases_gradients.append(bias_grad)#.div_(self.beta))

        return weights_gradients, biases_gradients

    def energy(self, input=None,hidden=None, targets=None, batch_size=None):
        """
        Compute the total energy of the network.
        
        Args:
            states (list): List of state tensors for each layer [batch_size x layer_size]
            inputs (torch.Tensor): Input data [batch_size x input_size]
            targets (torch.Tensor, optional): Target data [batch_size x output_size]
        
        Returns:
            torch.Tensor: Total energy of the network averaged over batch
        """
        energy = 0.0
        states = [torch.rand(size, device=input.device) for size in self.layer_sizes]
        if input is not None:
            self.states[0] = input
        
        if hidden is not None:
            for i in range(1, len(self.layer_sizes)-1):
                self.states[i] = hidden[i-1]

        if targets is not None:
            self.states[-1] = targets


        
        # Calculate the energy according to the new formulation
        # squared_norm: for each layer, sum of squares of each row, then sum over layers.
        squared_norm = sum([(s * s).sum(dim=1) for s in states]) / 2.0

        # linear_terms: for each layer, compute dot(layer, bias).
        linear_terms = -sum([torch.sum(states[i+1] * self.biases[i], dim=1) for i in range(len(self.biases))])

        # quadratic_terms: for each adjacent pair of layers.
        quadratic_terms = -sum([
            ((states[i] @ self.weights[i]) * states[i+1]).sum(dim=1)
            for i in range(len(self.weights))
        ])

        energy = squared_norm + linear_terms + quadratic_terms
            
        return energy
    
    def cost(self, output, target, beta=0, grad=True):
        if grad:
            target = nn.functional.one_hot(target, num_classes=output.shape[1])
            target = target.float()
            output = output.float()
            return beta * (target - output)
        else:
            target = nn.functional.one_hot(target, num_classes=output.shape[1])
            target = target.float()
            output = output.float()
            return 0.5 * torch.mean(torch.sum((output - target) ** 2, dim=1)).item()
    
    def activation(self, x, grad=False):
        if grad:
            # Calculate the gradient of the clamping function
            return torch.where((x > 0) & (x < 1), torch.ones_like(x), torch.zeros_like(x))
        elif not grad:
            return torch.clamp(x, min=0, max=1)
        
    def pi(self, x):
        # Calculate the pi function for x
        return torch.clamp(x, min=0, max=1)
    
    def negative(self, input, target=None, beta=0):
        batch_size = input.shape[0]
        
        # Initialize states with proper batch dimension if not done
        if self.states is None or self.states[0].shape[0] != batch_size:
            self.states = [torch.rand(batch_size, size, device=input.device) for size in self.layer_sizes]
          
        # Clamp input
        input = input.reshape(batch_size, -1)
        self.states[0] = input

        with torch.inference_mode(): 
            # Fixed point iterations
            for step in range(self.n_steps):
                
                # Update hidden layers
                for i in range(1, len(self.layer_sizes)-1):
                    # Calculate values for equation components
                    a_prev = self.activation(self.states[i-1])
                    a_next = self.activation(self.states[i+1])
                    
                    feedforward = a_prev @ self.weights[i-1]
                    feedback = a_next @ self.weights[i].T
                    bias_term = self.biases[i-1]
                    
                
                    # Calculate gradient
                    grad = self.activation(self.states[i], grad=True) * (feedforward + feedback + bias_term) - self.states[i]
                    if hasattr(self, 'debug') and self.debug:
                        print(f"  Gradient = σ'(s_{i}) * [feedforward + feedback + bias] - s_{i} = {grad[0, :5].detach().cpu().numpy()}")
                    
                    # Update state
                    self.states[i].add_(self.step_size * grad)   
                    # Do batch normalization on state
                    # self.states[i] = self.states[i] - self.states[i].mean(dim=0, keepdim=True)
                    # self.states[i] = self.states[i] / (self.states[i].std(dim=0, keepdim=True) + 1e-8)
            
                # Last layer update
                a_prev = self.activation(self.states[-2])
                feedforward = a_prev @ self.weights[-1]
                bias_term = self.biases[-1]
                
                grad = self.activation(self.states[-1], grad=True) * (feedforward + bias_term) - self.states[-1]
                
                # Cost gradient if target provided
                if target is not None:
                    cost_grad = self.cost(self.states[-1], target, beta=beta, grad=True)
                    grad = grad + cost_grad

                # Update last state
                self.states[-1] = self.states[-1] + self.step_size * grad
                        
        return [s.clone() for s in self.states]
    
    def train(self):
        """
        Set the model to training mode.
        """
        self.training_mode = True
        return self
    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.training_mode = False
        return self

    def forward(self, input, target=None, beta=0):
        if self.training_mode:
            if self.optimizer is None:
                raise ValueError("Optimizer not set. Please set optimizer before training.")
            
         
            self.optimizer.zero_grad()

            # First Energy Minimization without cost
            first_states = self.negative(input, target)
           

            # Second Energy Minimization with cost
            second_states = self.negative(input, target, beta=self.beta)
           
            weights_gradients, biases_gradients = self.gradient((first_states, second_states))
            
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i].grad = weights_gradients[i]
                self.biases[i].grad = biases_gradients[i]
                
            return second_states[-1]
        else:
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] Forward pass in evaluation mode")
            states = self.negative(input)
            return states[-1]
