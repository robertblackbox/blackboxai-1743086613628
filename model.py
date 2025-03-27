import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class NeuralNet(nn.Module):
    """
    A flexible neural network architecture that can be configured with
    different layer sizes and dropout rates.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Size of the input layer
            hidden_sizes (List[int]): List of sizes for hidden layers
            output_size (int): Size of the output layer
            dropout_rate (float): Dropout probability for regularization
        """
        super(NeuralNet, self).__init__()
        
        # Input validation
        if not isinstance(hidden_sizes, list) or not hidden_sizes:
            raise ValueError("hidden_sizes must be a non-empty list of integers")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        
        # Create the layers dynamically
        layers = []
        current_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(current_size, output_size))
        
        # Combine all layers
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
            
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        # Validate input shape
        if x.dim() != 2 or x.size(1) != self.network[0].in_features:
            raise ValueError(
                f"Expected input of shape (batch_size, {self.network[0].in_features}), "
                f"but got {tuple(x.shape)}"
            )
        
        return self.network(x)
    
    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)