

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple

from .base import BaseRewardModel

class MLPRewardModel(BaseRewardModel):
    """
    Multi-layer perceptron reward model.
    Maps state (and optional action) to reward values.
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dims: List[int] = [64, 64],
                activation: nn.Module = nn.ReLU,
                action_input: bool = False,
                action_dim: Optional[int] = None):
        """
        Initialize the MLP reward model.
        
        Args:
            input_dim: Dimension of the state input
            hidden_dims: List of hidden layer dimensions
            activation: Activation function class
            action_input: Whether to include action as input
            action_dim: Dimension of action input (if action_input is True)
        """
        super().__init__()
        
        self.action_input = action_input
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Calculate total input dimension
        total_input_dim = input_dim
        if action_input and action_dim is not None:
            total_input_dim += action_dim
        
        # Build MLP layers
        layers = []
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
            
        # Output layer (scalar reward)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, 
               state: torch.Tensor, 
               action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            action: Optional action tensor of shape (batch_size, action_dim)
            
        Returns:
            Reward values tensor of shape (batch_size, 1)
        """
        if self.action_input and action is not None:
            # Concatenate state and action
            inputs = torch.cat([state, action], dim=1)
        else:
            inputs = state
            
        return self.model(inputs)
        
    @classmethod
    def load(cls, path: str) -> 'MLPRewardModel':
        """
        Load model from a file with additional parameters.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path)
        model_args = checkpoint.get('model_args', {})
        
        # Create model with saved arguments
        model = cls(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
        
    def save(self, path: str) -> None:
        """
        Save model with additional parameters.
        
        Args:
            path: Path to save the model
        """
        model_args = {
            'input_dim': self.input_dim,
            'action_input': self.action_input,
            'action_dim': self.action_dim
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_args': model_args
        }, path)


class EnsembleRewardModel(BaseRewardModel):
    """
    Ensemble of reward models to provide uncertainty estimates.
    Helps with exploration and robustness.
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dims: List[int] = [64, 64],
                num_models: int = 5,
                activation: nn.Module = nn.ReLU,
                action_input: bool = False,
                action_dim: Optional[int] = None):
        """
        Initialize the ensemble reward model.
        
        Args:
            input_dim: Dimension of the state input
            hidden_dims: List of hidden layer dimensions
            num_models: Number of models in the ensemble
            activation: Activation function class
            action_input: Whether to include action as input
            action_dim: Dimension of action input (if action_input is True)
        """
        super().__init__()
        
        self.num_models = num_models
        self.input_dim = input_dim
        self.action_input = action_input
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            MLPRewardModel(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                action_input=action_input,
                action_dim=action_dim
            )
            for _ in range(num_models)
        ])
        
    def forward(self, 
               state: torch.Tensor, 
               action: Optional[torch.Tensor] = None,
               return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the ensemble reward model.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            action: Optional action tensor of shape (batch_size, action_dim)
            return_uncertainty: Whether to return uncertainty estimate
            
        Returns:
            Mean reward values and optionally uncertainty
        """
        # Get predictions from all models
        all_preds = []
        for model in self.models:
            pred = model(state, action)
            all_preds.append(pred)
            
        # Stack predictions
        stacked_preds = torch.stack(all_preds, dim=0)  # (num_models, batch_size, 1)
        
        # Calculate mean prediction
        mean_pred = torch.mean(stacked_preds, dim=0)  # (batch_size, 1)
        
        if return_uncertainty:
            # Calculate uncertainty as standard deviation
            uncertainty = torch.std(stacked_preds, dim=0)  # (batch_size, 1)
            return mean_pred, uncertainty
        else:
            return mean_pred
            
    def save(self, path: str) -> None:
        """
        Save ensemble model.
        
        Args:
            path: Path to save the model
        """
        model_args = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_models': self.num_models,
            'action_input': self.action_input,
            'action_dim': self.action_dim
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_args': model_args
        }, path)
        
    @classmethod
    def load(cls, path: str) -> 'EnsembleRewardModel':
        """
        Load ensemble model from a file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path)
        model_args = checkpoint.get('model_args', {})
        
        # Create model with saved arguments
        model = cls(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
