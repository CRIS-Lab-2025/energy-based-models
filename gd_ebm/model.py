import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
import json
from pathlib import Path

class EnergyBasedModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, optimizer, beta=0.1, dt=0.1, n_steps=20, debug=False):
        """
        Initialize an Energy-Based Model with equilibrium propagation.
        
        Args:
            input_size (int): Size of input layer
            hidden_sizes (list): List of hidden layer sizes
            optimizer: PyTorch optimizer (will be set separately)
            beta (float): Influence of the output cost on the energy
            dt (float): Time step for dynamics
            n_steps (int): Number of steps for reaching equilibrium
            debug (bool): Whether to enable debugging mode
        """
        super().__init__()
        
        self.layer_sizes = [input_size] + hidden_sizes
        self.beta = beta
        self.dt = dt
        self.n_steps = n_steps
        self.training_mode = False
        self.debug = debug
        self.debug_log = {}
        
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
        self.optimizer = None
        
        # Debug log directory
        self.log_dir = Path('debug_logs')
        if self.debug:
            self.log_dir.mkdir(exist_ok=True)
            self.debug_iteration = 0
    
    def enable_debug(self):
        """Enable debug mode"""
        self.debug = True
        self.log_dir = Path('debug_logs')
        self.log_dir.mkdir(exist_ok=True)
        self.debug_iteration = 0
        print("Debug mode enabled. Logs will be saved to:", self.log_dir.absolute())
        
    def disable_debug(self):
        """Disable debug mode"""
        self.debug = False
        print("Debug mode disabled.")
    
    def reset_debug_log(self):
        """Reset the debug log"""
        self.debug_log = {}
        self.debug_iteration = 0
    
    def log_state(self, name, value, step=None):
        """Log a state value for debugging"""
        if not self.debug:
            return
            
        if isinstance(value, torch.Tensor):
            # Convert tensor to numpy for logging
            value_np = value.detach().cpu().numpy()
            
            # Handle different tensor dimensions
            if value_np.ndim == 0:  # scalar
                value_to_log = float(value_np)
            elif value_np.ndim == 1:  # vector
                if len(value_np) > 10:  # if vector is long, log summary stats
                    value_to_log = {
                        'mean': float(np.mean(value_np)),
                        'std': float(np.std(value_np)),
                        'min': float(np.min(value_np)),
                        'max': float(np.max(value_np)),
                        'shape': list(value_np.shape)
                    }
                else:
                    value_to_log = value_np.tolist()
            else:  # matrix or higher
                value_to_log = {
                    'mean': float(np.mean(value_np)),
                    'std': float(np.std(value_np)),
                    'min': float(np.min(value_np)),
                    'max': float(np.max(value_np)),
                    'shape': list(value_np.shape)
                }
        else:
            value_to_log = value
            
        if step is None:
            step = self.debug_iteration
            
        if name not in self.debug_log:
            self.debug_log[name] = {}
            
        self.debug_log[name][step] = value_to_log
    
    def save_debug_log(self, prefix='debug'):
        """Save the debug log to a file"""
        if not self.debug or not self.debug_log:
            return
            
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filepath = self.log_dir / f"{prefix}_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
            
        print(f"Debug log saved to {filepath}")
        
        # Reset log after saving
        self.reset_debug_log()
    
    def gradient(self, inputs):
        first_states, second_states = inputs
        
        weights_gradients = []
        biases_gradients = []
        
        if self.debug:
            print(f"\n--- Computing Gradients (Iteration {self.debug_iteration}) ---")
            self.log_state("gradient_inputs/first_states", [s.mean().item() for s in first_states])
            self.log_state("gradient_inputs/second_states", [s.mean().item() for s in second_states])
        
        for i in range(len(self.weights)):
            # Compute batch-wise outer products and take mean over batch dimension
            weight_grad = (torch.bmm(second_states[i+1].unsqueeze(2), second_states[i].unsqueeze(1)) - 
                         torch.bmm(first_states[i+1].unsqueeze(2), first_states[i].unsqueeze(1))).mean(0) / self.beta
            weights_gradients.append(weight_grad)
            
            bias_grad = (second_states[i+1] - first_states[i+1]).mean(dim=0)
            biases_gradients.append(bias_grad)
            
            if self.debug:
                print(f"  Layer {i}: Weight grad stats - Mean: {weight_grad.mean().item():.6f}, "
                      f"Std: {weight_grad.std().item():.6f}, "
                      f"Min: {weight_grad.min().item():.6f}, "
                      f"Max: {weight_grad.max().item():.6f}")
                print(f"  Layer {i}: Bias grad stats   - Mean: {bias_grad.mean().item():.6f}, "
                      f"Std: {bias_grad.std().item():.6f}, "
                      f"Min: {bias_grad.min().item():.6f}, "
                      f"Max: {bias_grad.max().item():.6f}")
                
                self.log_state(f"gradient/weight_{i}", weight_grad)
                self.log_state(f"gradient/bias_{i}", bias_grad)

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
        
        weight_energies = []
        bias_energies = []
        saturation_energies = []
        
        # Symmetric weight contributions
        for i in range(len(self.weights)):
            # Batch matrix multiplication
            weight_energy = -torch.mean(torch.sum(states[i] @ self.weights[i] * states[i+1], dim=1))
            energy += weight_energy
            weight_energies.append(weight_energy.item())
            
            # Add bias terms
            bias_energy = -torch.mean(torch.sum(states[i+1] * self.biases[i], dim=1))
            energy += bias_energy
            bias_energies.append(bias_energy.item())
            
            # Saturation cost (using soft bounds)
            sat_energy = torch.mean(torch.sum(F.softplus(states[i+1]) + F.softplus(-states[i+1]), dim=1))
            energy += sat_energy
            saturation_energies.append(sat_energy.item())
        
        # Add supervised cost if targets are provided
        target_energy = 0.0
        if targets is not None:
            target_energy = self.beta * 0.5 * torch.mean(torch.sum((states[-1] - targets) ** 2, dim=1))
            energy += target_energy
        
        if self.debug:
            print(f"\n--- Energy Calculation (Iteration {self.debug_iteration}) ---")
            print(f"  Input clamping energy: {input_energy.item():.6f}")
            for i in range(len(self.weights)):
                print(f"  Layer {i}: Weight energy: {weight_energies[i]:.6f}, "
                      f"Bias energy: {bias_energies[i]:.6f}, "
                      f"Saturation energy: {saturation_energies[i]:.6f}")
            if targets is not None:
                print(f"  Target energy (beta={self.beta}): {target_energy.item():.6f}")
            print(f"  Total energy: {energy.item():.6f}")
            
            self.log_state("energy/input", input_energy.item())
            self.log_state("energy/weight", weight_energies)
            self.log_state("energy/bias", bias_energies)
            self.log_state("energy/saturation", saturation_energies)
            if targets is not None:
                self.log_state("energy/target", target_energy.item())
            self.log_state("energy/total", energy.item())
            
        return energy
    
    def cost(self, output, target, beta=0, grad=True):
        if grad:
            cost_value = beta * (output - target)
            if self.debug:
                self.log_state("cost/grad", cost_value)
            return cost_value
        else:
            cost_value = 0.5 * torch.mean(torch.sum((output - target) ** 2, dim=1)).item()
            if self.debug:
                self.log_state("cost/value", cost_value)
            return cost_value
    
    def activation(self, x):
        activated = torch.sigmoid(x)
        if self.debug:
            self.log_state("activation/pre", x)
            self.log_state("activation/post", activated)
        return activated
    
    def negative(self, input, target=None, beta=0):
        batch_size = input.shape[0]
        
        # Initialize states with proper batch dimension if not done
        if self.states is None or self.states[0].shape[0] != batch_size:
            self.states = [torch.zeros(batch_size, size, device=input.device) for size in self.layer_sizes]
            
        # Clamp input
        self.states[0] = input.clone()
        
        if self.debug:
            print(f"\n--- Negative Phase (Iteration {self.debug_iteration}, Beta={beta}) ---")
            print(f"  Input shape: {input.shape}, Batch size: {batch_size}")
            print(f"  Target: {'Provided' if target is not None else 'None'}")
            self.log_state("negative/input", input)
            if target is not None:
                self.log_state("negative/target", target)
            
            # Log initial states
            for i, state in enumerate(self.states):
                self.log_state(f"negative/initial_state_{i}", state)
        
        # Fixed point iterations
        for step in range(self.n_steps):
            old_states = [s.clone() for s in self.states]
            
            for i in range(1, len(self.layer_sizes)):
                pre_activation = self.states[i-1] @ self.weights[i-1] + self.biases[i-1]
                
                if beta != 0 and i == len(self.layer_sizes)-1 and target is not None:
                    cost_grad = self.cost(self.states[i], target, beta, grad=True)
                    pre_activation += cost_grad
                    
                    if self.debug and step == 0:
                        print(f"  Adding cost gradient to layer {i}, magnitude: {cost_grad.abs().mean().item():.6f}")
                
                self.states[i] = self.activation(pre_activation)
            
            # Calculate state changes for convergence monitoring
            if self.debug:
                state_changes = [torch.mean(torch.abs(self.states[i] - old_states[i])).item() 
                              for i in range(1, len(self.states))]
                avg_change = sum(state_changes) / len(state_changes)
                
                if step % 5 == 0 or step == self.n_steps - 1:
                    print(f"  Step {step+1}/{self.n_steps}: Avg state change: {avg_change:.6f}")
                    
                self.log_state(f"negative/step_{step}/changes", state_changes)
                
                # Log states periodically
                if step % 5 == 0 or step == self.n_steps - 1:
                    for i, state in enumerate(self.states):
                        self.log_state(f"negative/step_{step}/state_{i}", state)
        
        if self.debug:
            # Calculate final energy
            energy = self.energy(self.states, input, target if beta > 0 else None)
            print(f"  Final energy: {energy.item():.6f}")
            
            # Log final states
            for i, state in enumerate(self.states):
                self.log_state(f"negative/final_state_{i}", state)
        
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
            
            if self.debug:
                self.debug_iteration += 1
                print(f"\n============ Forward Pass (Iteration {self.debug_iteration}) ============")
                print(f"Input shape: {input.shape}, Target: {'Provided' if target is not None else 'None'}")
                print(f"Model state: {'Training' if self.training_mode else 'Evaluation'}")
                self.log_state("forward/input", input)
                if target is not None:
                    self.log_state("forward/target", target)
            
            self.optimizer.zero_grad()

            # First Energy Minimization without cost
            if self.debug:
                print("\nRunning first phase (free phase) with beta=0")
            first_states = self.negative(input, target)

            # Second Energy Minimization with cost
            if self.debug:
                print(f"\nRunning second phase (clamped phase) with beta={self.beta}")
            second_states = self.negative(input, target, beta=self.beta)

            # Compute gradients
            if self.debug:
                print("\nComputing weight and bias gradients from equilibrium states")
            weights_gradients, biases_gradients = self.gradient((first_states, second_states))

            # Update weights and biases
            for i in range(len(self.weights)):
                self.weights[i].grad = weights_gradients[i].T
                self.biases[i].grad = biases_gradients[i]
                
                if self.debug:
                    weight_grad_norm = torch.norm(self.weights[i].grad).item()
                    bias_grad_norm = torch.norm(self.biases[i].grad).item()
                    print(f"  Layer {i}: Weight grad norm: {weight_grad_norm:.6f}, "
                          f"Bias grad norm: {bias_grad_norm:.6f}")
                    self.log_state(f"forward/weight_grad_norm_{i}", weight_grad_norm)
                    self.log_state(f"forward/bias_grad_norm_{i}", bias_grad_norm)

            if self.debug:
                print("\nApplying optimizer step")
                # Log parameter stats before update
                for i, w in enumerate(self.weights):
                    self.log_state(f"forward/pre_update_weight_{i}", w)
                for i, b in enumerate(self.biases):
                    self.log_state(f"forward/pre_update_bias_{i}", b)
                
            self.optimizer.step()
            
            if self.debug:
                # Log parameter stats after update
                for i, w in enumerate(self.weights):
                    self.log_state(f"forward/post_update_weight_{i}", w)
                for i, b in enumerate(self.biases):
                    self.log_state(f"forward/post_update_bias_{i}", b)
                    
                # Calculate weight change magnitude
                for i in range(len(self.weights)):
                    pre = self.debug_log[f"forward/pre_update_weight_{i}"][self.debug_iteration]
                    post = self.debug_log[f"forward/post_update_weight_{i}"][self.debug_iteration]
                    if isinstance(pre, dict) and 'mean' in pre:
                        change = post['mean'] - pre['mean']
                    else:
                        change = np.mean(np.array(post) - np.array(pre))
                    print(f"  Layer {i}: Avg weight change: {change:.8f}")
                
                print("\nForward pass complete")
                
                # Log output
                self.log_state("forward/output", second_states[-1])
                
                # Save logs periodically
                if self.debug_iteration % 10 == 0:
                    self.save_debug_log(prefix=f"iter_{self.debug_iteration}")
            
            return second_states[-1]
        else:
            # Evaluation mode - just do one forward pass without cost
            if self.debug:
                self.debug_iteration += 1
                print(f"\n============ Evaluation Pass (Iteration {self.debug_iteration}) ============")
                print(f"Input shape: {input.shape}")
                self.log_state("eval/input", input)
                
            states = self.negative(input)
            
            if self.debug:
                self.log_state("eval/output", states[-1])
                if self.debug_iteration % 10 == 0:
                    self.save_debug_log(prefix=f"eval_{self.debug_iteration}")
                    
            return states[-1]
