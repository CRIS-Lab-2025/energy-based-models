import torch
import torch.nn.functional as F
from model._network import Network
from util.config import Config

class ConvolutionalNetwork(Network):
    """
    A convolutional network implementation for energy-based models.
    
    This network uses convolutional layers with the specified kernel sizes
    and channel configurations. It is designed to work with the ConvHopfieldEnergy
    energy function.
    """
    def __init__(self, config: Config):
        """
        Initialize a convolutional network.
        
        Args:
            config: Configuration object with the following keys:
                - model.channels: List of channel dimensions for each layer
                - model.kernel_size: Kernel size for convolutions
                - model.input_shape: Tuple of (channels, height, width) for input
                - training.batch_size: Batch size for training
        """
        self.channels = config.model.get("channels", [1, 16, 32, 1])
        self.kernel_size = config.model.get("kernel_size", 3)
        self.input_shape = config.model.get("input_shape", (1, 28, 28))
        self.activation = config.model.get("activation", "hard-sigmoid")
        batch_size = config.training["batch_size"]
        
        # Calculate total number of neurons
        total_neurons = sum(c * self.input_shape[1] * self.input_shape[2] for c in self.channels)
        
        # Initialize the base network
        super().__init__(config, total_neurons, batch_size)
        
        # Override the state with appropriate shapes for convolution
        self._state = self.config.to_device(torch.zeros(
            batch_size, 
            self.channels[-1], 
            self.input_shape[1], 
            self.input_shape[2]
        ))
        
        # Create convolutional weights
        self._conv_weights = []
        for i in range(len(self.channels) - 1):
            weight = torch.nn.Parameter(
                self.config.to_device(torch.zeros(
                    self.channels[i+1], 
                    self.channels[i], 
                    self.kernel_size, 
                    self.kernel_size
                ))
            )
            self._conv_weights.append(weight)
        
        # Create biases for each channel in each layer
        self._conv_biases = []
        for i in range(len(self.channels)):
            bias = torch.nn.Parameter(
                self.config.to_device(torch.zeros(self.channels[i]))
            )
            self._conv_biases.append(bias)
        
        # Define free layers (all neurons except input layer)
        # This is needed for the FixedPointUpdater
        self.free_layers = list(range(self.channels[0] * self.input_shape[1] * self.input_shape[2], total_neurons))
        
        # Define layers structure for compatibility with FixedPointUpdater
        # Each layer is a list of neuron indices
        self.layers = []
        start_idx = 0
        for i, c in enumerate(self.channels):
            layer_size = c * self.input_shape[1] * self.input_shape[2]
            self.layers.append(list(range(start_idx, start_idx + layer_size)))
            start_idx += layer_size
        
        # Initialize weights using the specified distribution
        self._init_weights()
        
        # Create a proper convolutional weight tensor for the energy function
        self._weights = self._create_conv_weight_tensor()
        
        # Create a proper bias tensor
        self._biases = self._create_bias_tensor()
    
    def _create_conv_weight_tensor(self):
        """
        Create a proper convolutional weight tensor for the energy function.
        This ensures the weights have the right shape for convolution operations.
        """
        # For each channel, create a kernel tensor
        kernels = []
        for i, c in enumerate(self.channels):
            if i < len(self.channels) - 1:  # Skip the last channel as it doesn't have outgoing connections
                kernel = self._conv_weights[i]
                kernels.append(kernel)
        
        # If there are no kernels, return an empty tensor
        if not kernels:
            return torch.nn.Parameter(torch.tensor([]))
        
        # Combine all kernels into a single parameter
        return torch.nn.Parameter(torch.cat([k.reshape(1, -1) for k in kernels], dim=1))
    
    def _create_bias_tensor(self):
        """
        Create a proper bias tensor for the energy function.
        """
        # Combine all biases into a single parameter
        return torch.nn.Parameter(torch.cat([b for b in self._conv_biases]))
    
    def _init_weights(self):
        """Initialize weights using the distribution specified in the config."""
        init_method = self.config.model.get("weight_init_dist", "xavier_uniform")
        
        for weights in self._conv_weights:
            if init_method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(weights)
            elif init_method == "xavier_normal":
                torch.nn.init.xavier_normal_(weights)
            elif init_method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(weights)
            elif init_method == "kaiming_normal":
                torch.nn.init.kaiming_normal_(weights)
    
    def set_input(self, input: torch.Tensor):
        """
        Set the input layer of the network.
        
        Args:
            input: Tensor of shape (batch_size, channels, height, width)
        """
        input = self.config.to_device(input)
        
        # Ensure input has the right shape
        if input.dim() == 2:
            # Reshape flat input to match expected dimensions
            input = input.view(input.size(0), self.input_shape[0], 
                              self.input_shape[1], self.input_shape[2])
        
        # Update the state with the input
        self._state[:input.shape[0]] = input
        
        return self._state, self._weights
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor after passing through all layers
        """
        # Ensure input is on the correct device
        x = self.config.to_device(x)
        
        # Pass through each convolutional layer
        for i in range(len(self._conv_weights)):
            x = F.conv2d(x, self._conv_weights[i], self._conv_biases[i], padding='same')
            
            # Apply activation function except for the output layer
            if i < len(self._conv_weights) - 1:
                if self.activation == "sigmoid":
                    x = torch.sigmoid(x)
                elif self.activation == "tanh":
                    x = torch.tanh(x)
                elif self.activation == "relu":
                    x = F.relu(x)
                elif self.activation == "hard-sigmoid":
                    x = torch.clamp(0.2 * x + 0.5, 0.0, 1.0)
        
        return x
    
    def get_clamped_indices(self):
        """
        Get indices of clamped nodes (input layer).
        
        Returns:
            List of indices corresponding to input nodes
        """
        # In a convolutional network, typically the input layer is clamped
        # Return indices corresponding to the first channel's neurons
        return list(range(self.channels[0] * self.input_shape[1] * self.input_shape[2]))
    
    def reset_state(self):
        """Reset the network state to zeros."""
        self._state = self.config.to_device(torch.zeros_like(self._state))
        
    def clamp_weights(self, min_val=-1.0, max_val=1.0):
        """
        Clamp weights to the specified range.
        
        Args:
            min_val: Minimum value for weights
            max_val: Maximum value for weights
        """
        for i in range(len(self._conv_weights)):
            self._conv_weights[i].data.clamp_(min_val, max_val) 