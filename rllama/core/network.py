from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """
    Base class for neural networks.
    
    This class provides a common interface for various neural network
    architectures used in reinforcement learning.
    """
    
    def __init__(self):
        """Initialize a neural network."""
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError
        
    def save(self, path: str) -> None:
        """
        Save the network to a file.
        
        Args:
            path: Path to save the network to
        """
        torch.save(self.state_dict(), path)
        
    def load(self, path: str, device: str = "cpu") -> None:
        """
        Load the network from a file.
        
        Args:
            path: Path to load the network from
            device: Device to load the network to
        """
        self.load_state_dict(torch.load(path, map_location=device))


class MLP(Network):
    """
    Multi-layer perceptron network.
    
    Attributes:
        layers: Sequential layers of the network
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 64],
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None,
    ):
        """
        Initialize an MLP.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            hidden_dims: Dimensions of hidden layers
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        super().__init__()
        
        # Build network
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation)
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation is not None:
            layers.append(output_activation)
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.layers(x)