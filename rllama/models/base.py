#!/usr/bin/env python3
"""
Base classes for reward models in RLlama framework.
Provides the foundation for all neural network reward models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import logging

class BaseRewardModel(nn.Module, ABC):
    """
    Abstract base class for all reward models.
    
    This class provides the foundation for neural network-based reward models
    that learn to predict reward values from states and optionally actions.
    """
    
    def __init__(self):
        """Initialize the base reward model."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def forward(self, 
                state: torch.Tensor, 
                action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Optional action tensor of shape (batch_size, action_dim)
            
        Returns:
            Reward values tensor of shape (batch_size, 1)
        """
        pass
        
    def predict(self, 
                state: Union[torch.Tensor, list], 
                action: Optional[Union[torch.Tensor, list]] = None,
                device: Optional[str] = None) -> torch.Tensor:
        """
        Convenience method for prediction with automatic tensor conversion.
        
        Args:
            state: State input (tensor or list/array)
            action: Optional action input (tensor or list/array)
            device: Device to run prediction on
            
        Returns:
            Predicted reward values
        """
        # Convert inputs to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        if action is not None and not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        # Move to device if specified
        if device is not None:
            state = state.to(device)
            if action is not None:
                action = action.to(device)
                
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action is not None and action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Set to evaluation mode
        was_training = self.training
        self.eval()
        
        try:
            with torch.no_grad():
                predictions = self.forward(state, action)
            return predictions
        finally:
            # Restore training mode
            self.train(was_training)
            
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'pytorch_version': torch.__version__
        }, path)
        
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None):
        """
        Load a model from a file.
        
        Args:
            path: Path to the saved model
            map_location: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=map_location)
        
        # This is a basic implementation - subclasses should override
        # to handle model-specific arguments
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
    def get_num_parameters(self) -> int:
        """Get the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
        
    def get_num_trainable_parameters(self) -> int:
        """Get the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze_parameters(self) -> None:
        """Freeze all parameters (disable gradients)."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters (enable gradients)."""
        for param in self.parameters():
            param.requires_grad = True
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'class_name': self.__class__.__name__,
            'total_parameters': self.get_num_parameters(),
            'trainable_parameters': self.get_num_trainable_parameters(),
            'device': next(self.parameters()).device.type,
            'dtype': next(self.parameters()).dtype,
            'training_mode': self.training
        }
        
    def __str__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (f"{info['class_name']}("
                f"params={info['total_parameters']}, "
                f"device={info['device']}, "
                f"training={info['training_mode']})")
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}()"
