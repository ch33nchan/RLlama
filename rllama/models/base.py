

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

class BaseRewardModel(nn.Module):
    """Base class for neural network reward models."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Returns:
            Reward values as a tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
        
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'BaseRewardModel':
        """
        Load a model from a file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
