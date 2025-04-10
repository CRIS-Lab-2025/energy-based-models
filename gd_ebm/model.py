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
        
        self.layer_sizes = [input_size] + hidden_sizes
        #print(self.layer_sizes)
        self.beta = beta
        self.dt = dt
        self.n_steps = n_steps
        self.step_size = dt / n_steps
        self.training_mode = False
        
        # Initialize weights and biases for symmetric connections
        # We use separate weights for forward and backward to maintain symmetry explicitly
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.randn(self.layer_sizes[i], self.layer_sizes[i+1]) / 
                torch.sqrt(torch.tensor(self.layer_sizes[i]))
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
        
        weights_gradients = []
        biases_gradients = []
        
        for i in range(len(self.weights)):
            # Compute batch-wise outer products and take mean over batch dimension
            weight_grad = (torch.bmm(second_states[i+1].unsqueeze(2), second_states[i].unsqueeze(1)) - 
                         torch.bmm(first_states[i+1].unsqueeze(2), first_states[i].unsqueeze(1))).mean(0) / self.beta
            weights_gradients.append(weight_grad)
            
            bias_grad = (second_states[i+1] - first_states[i+1]).mean(dim=0)
            biases_gradients.append(bias_grad)

        return weights_gradients, biases_gradients

    def energy(self, states, inputs, targets=None):
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
        
        # Input clamping energy
        input_energy = 0.5 * torch.mean(torch.sum((states[0] - inputs) ** 2, dim=1))
        energy += input_energy
        
        # Symmetric weight contributions
        for i in range(len(self.weights)):
            # Batch matrix multiplication
            weight_energy = -torch.mean(torch.sum(states[i] @ self.weights[i] * states[i+1], dim=1))
            energy += weight_energy
            
            # Add bias terms
            bias_energy = -torch.mean(torch.sum(states[i+1] * self.biases[i], dim=1))
            energy += bias_energy
            
            # # Saturation cost (using soft bounds)
            # sat_energy = torch.mean(torch.sum(F.softplus(states[i+1]) + F.softplus(-states[i+1]), dim=1))
            # energy += sat_energy
        
        # Add supervised cost if targets are provided
        if targets is not None:
            target_energy = self.beta * 0.5 * torch.mean(torch.sum((states[-1] - targets) ** 2, dim=1))
            energy += target_energy
            
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
            # Calculate the gradient of the sigmoid function for x
            return torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif not grad:
            return torch.sigmoid(x)
    
    def negative(self, input, target=None, beta=0):
        batch_size = input.shape[0]
        
        # Initialize states with proper batch dimension if not done
        if self.states is None or self.states[0].shape[0] != batch_size:
            self.states = [torch.rand(batch_size, size, device=input.device) for size in self.layer_sizes]
       
        # for i in range(len(self.layer_sizes)):
        #     #print(f"Layer {i}: {self.layer_sizes[i]} -> {self.layer_sizes[i+1]}")
        #     # PRint state shapes
        #     print(f"State {i}: {self.states[i].shape}")
            
        
        # Clamp input
        input = input.reshape(batch_size, -1)
        self.states[0] = input.clone()
        
        # Fixed point iterations
        for step in range(self.n_steps):
            for i in range(1, len(self.layer_sizes)-1):
                grad = self.activation(self.states[i], grad=True)*(self.activation(self.states[i-1]) @ self.weights[i-1] + self.activation(self.states[i+1]) @ self.weights[i].T + self.biases[i-1]) - self.states[i]
                # Update state with gradient descent
                self.states[i] = self.states[i] - self.step_size * grad                

            # Last layer
            grad = self.activation(self.states[-1], grad=True)*(self.activation(self.states[-2]) @ self.weights[-1] + self.biases[-1]) - self.states[-1]
            cost_grad = self.cost(self.states[-1], target, beta=beta, grad=True) if target is not None else 0
            # Update state with gradient descent
            self.states[-1] = self.states[-1] - self.step_size * (grad + cost_grad)
                
                
                        
        return [s.clone() for s in self.states]
    
    def train(self):
        self.training_mode = True
        return self
    
    def eval(self):
        self.training_mode = False
        return self
    
    def forward(self, input, target=None, beta=0):
        if self.training_mode:
            if self.optimizer is None:
                raise ValueError("Optimizer not set. Please set optimizer before training.")
            #print(input.shape)
            self.optimizer.zero_grad()

            # First Energy Minimization without cost
            first_states = self.negative(input, target)

            # Second Energy Minimization with cost
            second_states = self.negative(input, target, beta=self.beta)

            # Compute gradients
            weights_gradients, biases_gradients = self.gradient((first_states, second_states))
            # Print gradients
            print("Weights Gradients:")
            for i, grad in enumerate(weights_gradients):
                print(f"Layer {i}: {grad}")
            print("Biases Gradients:")
            for i, grad in enumerate(biases_gradients):
                print(f"Layer {i}: {grad}")
            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i].grad = weights_gradients[i].T
                self.biases[i].grad = biases_gradients[i]

            self.optimizer.step()
            
            return second_states[-1]
        else:
            # Evaluation mode - just do one forward pass without cost
            states = self.negative(input)
            return states[-1]
