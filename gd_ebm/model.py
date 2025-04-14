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
        self.step_size = dt 
        self.training_mode = False
        self.debug = False
        # Initialize weights and biases for symmetric connections
        # We use separate weights for forward and backward to maintain symmetry explicitly
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.ones(self.layer_sizes[i], self.layer_sizes[i+1]) / 
                torch.sqrt(torch.tensor(4))
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

            weight_grad = (torch.bmm(second_states[i].unsqueeze(2),second_states[i+1].unsqueeze(1)) - 
                         torch.bmm(first_states[i].unsqueeze(2), first_states[i+1].unsqueeze(1))).mean(0) / self.beta
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
        
    def pi(self, x):
        # Calculate the pi function for x
        return torch.clamp(x, min=0, max=1)
    
    def negative(self, input, target=None, beta=0):
        batch_size = input.shape[0]
        
        # Initialize states with proper batch dimension if not done
        if self.states is None or self.states[0].shape[0] != batch_size:
            self.states = [torch.ones(batch_size, size, device=input.device) for size in self.layer_sizes]
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Initialized states with batch size: {batch_size}")
                for i, state in enumerate(self.states):
                    print(f"[DEBUG] Initial state[{i}] shape: {state.shape}, mean: {state.mean().item():.4f}")
                    print(f"[DEBUG] Initial state[{i}] values:\n{state[0].detach().cpu().numpy()}")
        
        # Clamp input
        input = input.reshape(batch_size, -1)
        self.states[0] = input.clone()
        if hasattr(self, 'debug') and self.debug:
            print(f"[DEBUG] Input clamped to state[0], shape: {self.states[0].shape}")
            print(f"[DEBUG] Input values:\n{self.states[0][0].detach().cpu().numpy()}")
            
            print("\n[DEBUG] Network weights:")
            for i, w in enumerate(self.weights):
                print(f"[DEBUG] Weight[{i}] shape: {w.shape}, mean: {w.mean().item():.4f}")
                print(f"[DEBUG] Weight[{i}] values:\n{w.detach().cpu().numpy()}")
            
            print("\n[DEBUG] Network biases:")
            for i, b in enumerate(self.biases):
                print(f"[DEBUG] Bias[{i}] shape: {b.shape}, mean: {b.mean().item():.4f}")
                print(f"[DEBUG] Bias[{i}] values:\n{b.detach().cpu().numpy()}")
            
            print(f"\n[DEBUG] Starting {self.n_steps} fixed point iterations with dt={self.dt}, step_size={self.step_size}")
        
        # Fixed point iterations
        for step in range(self.n_steps):
            if hasattr(self, 'debug') and self.debug:
                print(f"\n[DEBUG] Iteration {step+1}/{self.n_steps}")
            
            # Update hidden layers
            for i in range(1, len(self.layer_sizes)-1):
                # Calculate values for equation components
                a_current = self.activation(self.states[i])
                a_prev = self.activation(self.states[i-1])
                a_next = self.activation(self.states[i+1])
                
                feedforward = a_prev @ self.weights[i-1]
                feedback = a_next @ self.weights[i].T
                bias_term = self.biases[i-1]
                
                if hasattr(self, 'debug') and self.debug:
                    print(f"[DEBUG] Layer {i} update:")
                    print(f"  σ(s_{i}) = {a_current[0, :5].detach().cpu().numpy()} (showing first 5 of layer {i})")
                    print(f"  Full state[{i}]:\n{self.states[i][0].detach().cpu().numpy()}")
                    print(f"  σ'(s_{i}) = {self.activation(self.states[i], grad=True)[0, :5].detach().cpu().numpy()}")
                    print(f"  Feedforward: σ(s_{i-1}) @ W_{i-1} = {feedforward[0, :5].detach().cpu().numpy()}")
                    print(f"  Feedback: σ(s_{i+1}) @ W_{i}^T = {feedback[0, :5].detach().cpu().numpy()}")
                
                # Calculate gradient
                grad = self.activation(self.states[i], grad=True) * (feedforward + feedback + bias_term) - self.states[i]
                if hasattr(self, 'debug') and self.debug:
                    print(f"  Gradient = σ'(s_{i}) * [feedforward + feedback + bias] - s_{i} = {grad[0, :5].detach().cpu().numpy()}")
                
                # Update state
                old_state = self.states[i].clone()
                self.states[i] = self.states[i] - self.step_size * grad 
                # Do batch normalization on state
                # self.states[i] = self.states[i] - self.states[i].mean(dim=0, keepdim=True)
                # self.states[i] = self.states[i] / (self.states[i].std(dim=0, keepdim=True) + 1e-8)

                if hasattr(self, 'debug') and self.debug:
                    print(f"  Updated state[{i}]:\n{self.states[i][0].detach().cpu().numpy()}")
                    print(f"  State update: s_{i} = s_{i} - {self.step_size} * grad")
                    print(f"  State change: {(self.states[i] - old_state)[0, :5].detach().cpu().numpy()}")
                    print(f"  Updated state[{i}]:\n{self.states[i][0].detach().cpu().numpy()}")

            # Last layer update
            a_last = self.activation(self.states[-1])
            a_prev = self.activation(self.states[-2])
            feedforward = a_prev @ self.weights[-1]
            bias_term = self.biases[-1]
            
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Last layer update:")
                print(f"  σ(s_last) = {a_last[0, :5].detach().cpu().numpy()} (showing first 5 of output layer)")
                print(f"  Full last state:\n{self.states[-1][0].detach().cpu().numpy()}")
                print(f"  Feedforward: σ(s_prev) @ W_last = {feedforward[0, :5].detach().cpu().numpy()}")
            
            grad = self.activation(self.states[-1], grad=True) * (feedforward + bias_term) - self.states[-1]
            if hasattr(self, 'debug') and self.debug:
                print(f"  Base gradient = {grad[0, :5].detach().cpu().numpy()}")
            
            # Cost gradient if target provided
            if target is not None:
                cost_grad = self.cost(self.states[-1], target, beta=beta, grad=True)
                if hasattr(self, 'debug') and self.debug:
                    print(f"  Target: {target[:5].detach().cpu().numpy() if torch.is_tensor(target) else target[:5]}")
                    print(f"  Cost gradient (β={beta}): {cost_grad[0, :5].detach().cpu().numpy()}")
                grad = grad + cost_grad
                if hasattr(self, 'debug') and self.debug:
                    print(f"  Total gradient = base_grad + cost_grad = {grad[0, :5].detach().cpu().numpy()}")
            
            # Update last state
            old_state = self.states[-1].clone()
            self.states[-1] = self.states[-1] - self.step_size * grad
            if hasattr(self, 'debug') and self.debug:
                print(f"  Last state update: change = {(self.states[-1] - old_state)[0, :5].detach().cpu().numpy()}")
                print(f"  Updated last state:\n{self.states[-1][0].detach().cpu().numpy()}")
            
            if hasattr(self, 'debug') and self.debug:
                if step == 0 or step == self.n_steps-1:
                    print(f"\n[DEBUG] Energy after iteration {step+1}: {self.energy(self.states, input, targets=target).item():.4f}")
                        
        return [s.clone() for s in self.states]

    def forward(self, input, target=None, beta=0):
        if self.training_mode:
            if self.optimizer is None:
                raise ValueError("Optimizer not set. Please set optimizer before training.")
            
            if hasattr(self, 'debug') and self.debug:
                print(f"\n[DEBUG] Forward pass in training mode (β={self.beta})")
            self.optimizer.zero_grad()

            # First Energy Minimization without cost
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] ===== FIRST PHASE (β=0) =====")
            first_states = self.negative(input, target)
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] First phase completed, final output: {first_states[-1][0, :5].detach().cpu().numpy()}")
                print(f"[DEBUG] Complete first phase final output:\n{first_states[-1][0].detach().cpu().numpy()}")

            # Second Energy Minimization with cost
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] ===== SECOND PHASE (β={self.beta}) =====")
            second_states = self.negative(input, target, beta=self.beta)
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Second phase completed, final output: {second_states[-1][0, :5].detach().cpu().numpy()}")
                print(f"[DEBUG] Complete second phase final output:\n{second_states[-1][0].detach().cpu().numpy()}")

            # Compute gradients
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] Computing parameter gradients:")
            weights_gradients, biases_gradients = self.gradient((first_states, second_states))
            
            # Print parameter gradients
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] Weight gradients (calculated as (s2⊗s2' - s1⊗s1')/β):")
                for i, grad in enumerate(weights_gradients):
                    print(f"  Layer {i}: shape {grad.shape}, mean {grad.mean().item():.4f}, std {grad.std().item():.4f}")
                    print(f"  Complete weight gradient[{i}]:\n{grad.detach().cpu().numpy()}")
                
                print("\n[DEBUG] Bias gradients (calculated as mean(s2' - s1')):")
                for i, grad in enumerate(biases_gradients):
                    print(f"  Layer {i}: shape {grad.shape}, mean {grad.mean().item():.4f}, std {grad.std().item():.4f}")
                    print(f"  Complete bias gradient[{i}]:\n{grad.detach().cpu().numpy()}")

            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i].grad = weights_gradients[i]
                self.biases[i].grad = biases_gradients[i]

            self.optimizer.step()
            
            # Print updated weights after optimization step
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] Updated weights after optimization step:")
                for i, w in enumerate(self.weights):
                    print(f"  Weight[{i}] mean: {w.mean().item():.4f}, std: {w.std().item():.4f}")
                    print(f"  Updated weight[{i}]:\n{w.detach().cpu().numpy()}")
                
                for i, b in enumerate(self.biases):
                    print(f"  Bias[{i}] mean: {b.mean().item():.4f}, std: {b.std().item():.4f}")
                    print(f"  Updated bias[{i}]:\n{b.detach().cpu().numpy()}")
                
            return second_states[-1]
        else:
            if hasattr(self, 'debug') and self.debug:
                print("\n[DEBUG] Forward pass in evaluation mode")
            states = self.negative(input)
            return states[-1]
