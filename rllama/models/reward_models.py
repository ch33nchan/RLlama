#!/usr/bin/env python3
"""
Neural network reward models for RLlama framework.
Provides MLP and ensemble models for learning reward functions from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import warnings

from .base import BaseRewardModel

class MLPRewardModel(BaseRewardModel):
    """
    Multi-layer perceptron reward model with advanced features.
    Maps state (and optional action) to reward values with configurable architecture.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int] = [128, 64, 32],
                 activation: Union[nn.Module, Callable] = nn.ReLU,
                 dropout_rate: float = 0.0,
                 batch_norm: bool = False,
                 output_activation: Optional[nn.Module] = None,
                 action_input: bool = False,
                 action_dim: Optional[int] = None,
                 layer_norm: bool = False,
                 residual_connections: bool = False):
        """
        Initialize the MLP reward model.
        
        Args:
            input_dim: Dimension of the state input
            hidden_dims: List of hidden layer dimensions
            activation: Activation function class or callable
            dropout_rate: Dropout rate (0.0 means no dropout)
            batch_norm: Whether to use batch normalization
            output_activation: Optional activation for output layer
            action_input: Whether to include action as input
            action_dim: Dimension of action input (if action_input is True)
            layer_norm: Whether to use layer normalization
            residual_connections: Whether to use residual connections
        """
        super().__init__()
        
        self.action_input = action_input
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual_connections = residual_connections
        
        # Calculate total input dimension
        total_input_dim = input_dim
        if action_input and action_dim is not None:
            total_input_dim += action_dim
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.layer_norms = nn.ModuleList() if layer_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None
        
        prev_dim = total_input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization layers
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
                
            # Dropout
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
            
        # Output layer (scalar reward)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Store activation function
        if isinstance(activation, type):
            self.activation = activation()
        else:
            self.activation = activation
            
        self.output_activation = output_activation
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.ELU)):
                # He initialization for ReLU-like activations
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else:
                # Xavier initialization for other activations
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # Initialize output layer with smaller weights
        nn.init.xavier_normal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
        
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
        # Prepare input
        if self.action_input and action is not None:
            # Concatenate state and action
            x = torch.cat([state, action], dim=1)
        else:
            x = state
            
        # Store input for potential residual connections
        residual = x
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            # Linear transformation
            x = layer(x)
            
            # Batch normalization
            if self.batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
                
            # Layer normalization
            if self.layer_norm and self.layer_norms is not None:
                x = self.layer_norms[i](x)
                
            # Activation
            x = self.activation(x)
            
            # Dropout
            if self.dropout_rate > 0 and self.dropouts is not None:
                x = self.dropouts[i](x)
                
            # Residual connection (if dimensions match)
            if (self.residual_connections and i > 0 and 
                x.shape[-1] == residual.shape[-1]):
                x = x + residual
                
            residual = x
            
        # Output layer
        x = self.output_layer(x)
        
        # Output activation
        if self.output_activation is not None:
            x = self.output_activation(x)
            
        return x
        
    def get_features(self, 
                    state: torch.Tensor, 
                    action: Optional[torch.Tensor] = None,
                    layer_idx: int = -1) -> torch.Tensor:
        """
        Extract features from a specific layer.
        
        Args:
            state: State tensor
            action: Optional action tensor
            layer_idx: Index of layer to extract features from (-1 for last hidden layer)
            
        Returns:
            Feature tensor from specified layer
        """
        # Prepare input
        if self.action_input and action is not None:
            x = torch.cat([state, action], dim=1)
        else:
            x = state
            
        # Forward through layers up to specified index
        target_layer = len(self.layers) + layer_idx if layer_idx < 0 else layer_idx
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)
            if self.layer_norm and self.layer_norms is not None:
                x = self.layer_norms[i](x)
                
            x = self.activation(x)
            
            if self.dropout_rate > 0 and self.dropouts is not None:
                x = self.dropouts[i](x)
                
            if i == target_layer:
                break
                
        return x
        
    def save(self, path: str) -> None:
        """
        Save model with comprehensive metadata.
        
        Args:
            path: Path to save the model
        """
        model_args = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation.__class__.__name__,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'action_input': self.action_input,
            'action_dim': self.action_dim,
            'layer_norm': self.layer_norm,
            'residual_connections': self.residual_connections
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_args': model_args,
            'pytorch_version': torch.__version__
        }, path)
        
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'MLPRewardModel':
        """
        Load model from file with backward compatibility.
        
        Args:
            path: Path to the saved model
            map_location: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=map_location)
        model_args = checkpoint.get('model_args', {})
        
        # Handle backward compatibility for activation
        if 'activation' in model_args and isinstance(model_args['activation'], str):
            activation_map = {
                'ReLU': nn.ReLU,
                'LeakyReLU': nn.LeakyReLU,
                'ELU': nn.ELU,
                'Tanh': nn.Tanh,
                'Sigmoid': nn.Sigmoid,
                'GELU': nn.GELU
            }
            model_args['activation'] = activation_map.get(model_args['activation'], nn.ReLU)
            
        # Create model with saved arguments
        model = cls(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class EnsembleRewardModel(BaseRewardModel):
    """
    Ensemble of reward models with uncertainty quantification.
    Provides robust predictions and uncertainty estimates for exploration.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int] = [128, 64, 32],
                 num_models: int = 5,
                 activation: Union[nn.Module, Callable] = nn.ReLU,
                 dropout_rate: float = 0.1,
                 batch_norm: bool = False,
                 action_input: bool = False,
                 action_dim: Optional[int] = None,
                 diversity_regularization: float = 0.0,
                 bootstrap_data: bool = True):
        """
        Initialize the ensemble reward model.
        
        Args:
            input_dim: Dimension of the state input
            hidden_dims: List of hidden layer dimensions
            num_models: Number of models in the ensemble
            activation: Activation function class or callable
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            action_input: Whether to include action as input
            action_dim: Dimension of action input (if action_input is True)
            diversity_regularization: Weight for diversity regularization loss
            bootstrap_data: Whether to use bootstrap sampling during training
        """
        super().__init__()
        
        self.num_models = num_models
        self.input_dim = input_dim
        self.action_input = action_input
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.diversity_regularization = diversity_regularization
        self.bootstrap_data = bootstrap_data
        
        # Create ensemble of models with slight architectural variations
        self.models = nn.ModuleList()
        
        for i in range(num_models):
            # Add slight variation to each model for diversity
            model_dropout = dropout_rate + (i * 0.01)  # Slightly different dropout rates
            
            # Optionally vary hidden dimensions slightly
            varied_hidden_dims = hidden_dims.copy()
            if len(varied_hidden_dims) > 0:
                # Add small random variation to first hidden layer
                variation = int(varied_hidden_dims[0] * 0.1)  # 10% variation
                varied_hidden_dims[0] += (i - num_models // 2) * (variation // num_models)
                varied_hidden_dims[0] = max(8, varied_hidden_dims[0])  # Ensure minimum size
            
            model = MLPRewardModel(
                input_dim=input_dim,
                hidden_dims=varied_hidden_dims,
                activation=activation,
                dropout_rate=model_dropout,
                batch_norm=batch_norm,
                action_input=action_input,
                action_dim=action_dim
            )
            
            self.models.append(model)
            
    def forward(self, 
                state: torch.Tensor, 
                action: Optional[torch.Tensor] = None,
                return_uncertainty: bool = False,
                return_all_predictions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the ensemble reward model.
        
        Args:
            state: State tensor of shape (batch_size, input_dim)
            action: Optional action tensor of shape (batch_size, action_dim)
            return_uncertainty: Whether to return uncertainty estimate
            return_all_predictions: Whether to return all individual predictions
            
        Returns:
            Mean reward values and optionally uncertainty and/or all predictions
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
        
        results = [mean_pred]
        
        if return_uncertainty:
            # Calculate uncertainty as standard deviation
            uncertainty = torch.std(stacked_preds, dim=0)  # (batch_size, 1)
            results.append(uncertainty)
            
        if return_all_predictions:
            results.append(stacked_preds)
            
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)
            
    def get_diversity_loss(self, 
                          state: torch.Tensor, 
                          action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate diversity regularization loss to encourage model diversity.
        
        Args:
            state: State tensor
            action: Optional action tensor
            
        Returns:
            Diversity loss tensor
        """
        if self.diversity_regularization <= 0:
            return torch.tensor(0.0, device=state.device)
            
        # Get predictions from all models
        all_preds = []
        for model in self.models:
            pred = model(state, action)
            all_preds.append(pred)
            
        # Stack predictions
        stacked_preds = torch.stack(all_preds, dim=0)  # (num_models, batch_size, 1)
        
        # Calculate pairwise distances between model predictions
        diversity_loss = 0.0
        count = 0
        
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                # L2 distance between predictions
                distance = torch.mean((stacked_preds[i] - stacked_preds[j]) ** 2)
                # We want to maximize diversity, so minimize negative distance
                diversity_loss -= distance
                count += 1
                
        if count > 0:
            diversity_loss = diversity_loss / count
            
        return self.diversity_regularization * diversity_loss
        
    def predict_with_confidence(self, 
                               state: torch.Tensor, 
                               action: Optional[torch.Tensor] = None,
                               confidence_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence estimates.
        
        Args:
            state: State tensor
            action: Optional action tensor
            confidence_threshold: Threshold for high confidence predictions
            
        Returns:
            Tuple of (predictions, uncertainties, confidence_mask)
        """
        with torch.no_grad():
            mean_pred, uncertainty = self.forward(state, action, return_uncertainty=True)
            
            # Create confidence mask (low uncertainty = high confidence)
            confidence_mask = uncertainty < confidence_threshold
            
        return mean_pred, uncertainty, confidence_mask
        
    def get_model_predictions(self, 
                             state: torch.Tensor, 
                             action: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get individual predictions from each model in the ensemble.
        
        Args:
            state: State tensor
            action: Optional action tensor
            
        Returns:
            List of prediction tensors from each model
        """
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(state, action)
                predictions.append(pred)
        return predictions
        
    def save(self, path: str) -> None:
        """
        Save ensemble model with metadata.
        
        Args:
            path: Path to save the model
        """
        model_args = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'num_models': self.num_models,
            'action_input': self.action_input,
            'action_dim': self.action_dim,
            'diversity_regularization': self.diversity_regularization,
            'bootstrap_data': self.bootstrap_data
        }
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_args': model_args,
            'pytorch_version': torch.__version__
        }, path)
        
    @classmethod
    def load(cls, path: str, map_location: Optional[str] = None) -> 'EnsembleRewardModel':
        """
        Load ensemble model from file.
        
        Args:
            path: Path to the saved model
            map_location: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=map_location)
        model_args = checkpoint.get('model_args', {})
        
        # Create model with saved arguments
        model = cls(**model_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

class BayesianRewardModel(BaseRewardModel):
    """
    Bayesian neural network reward model with built-in uncertainty quantification.
    Uses variational inference to learn weight distributions.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 activation: Union[nn.Module, Callable] = nn.ReLU,
                 prior_std: float = 1.0,
                 action_input: bool = False,
                 action_dim: Optional[int] = None):
        """
        Initialize Bayesian reward model.
        
        Args:
            input_dim: Dimension of state input
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            prior_std: Standard deviation of weight priors
            action_input: Whether to include action as input
            action_dim: Dimension of action input
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.action_input = action_input
        self.action_dim = action_dim
        self.prior_std = prior_std
        
        # Calculate total input dimension
        total_input_dim = input_dim
        if action_input and action_dim is not None:
            total_input_dim += action_dim
            
        # Build Bayesian layers
        self.layers = nn.ModuleList()
        prev_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layer = BayesianLinear(prev_dim, hidden_dim, prior_std=prior_std)
            self.layers.append(layer)
            prev_dim = hidden_dim
            
        # Output layer
        self.output_layer = BayesianLinear(prev_dim, 1, prior_std=prior_std)
        
        # Store activation
        if isinstance(activation, type):
            self.activation = activation()
        else:
            self.activation = activation
            
    def forward(self, 
                state: torch.Tensor, 
                action: Optional[torch.Tensor] = None,
                num_samples: int = 1) -> torch.Tensor:
        """
        Forward pass with sampling.
        
        Args:
            state: State tensor
            action: Optional action tensor
            num_samples: Number of samples to draw
            
        Returns:
            Sampled predictions
        """
        # Prepare input
        if self.action_input and action is not None:
            x = torch.cat([state, action], dim=1)
        else:
            x = state
            
        batch_size = x.shape[0]
        
        # Sample multiple times if requested
        if num_samples > 1:
            samples = []
            for _ in range(num_samples):
                sample = self._single_forward(x)
                samples.append(sample)
            return torch.stack(samples, dim=0)  # (num_samples, batch_size, 1)
        else:
            return self._single_forward(x)
            
    def _single_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass with weight sampling."""
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            
        x = self.output_layer(x)
        return x
        
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence for variational inference."""
        kl_div = 0.0
        for layer in self.layers:
            kl_div += layer.kl_divergence()
        kl_div += self.output_layer.kl_divergence()
        return kl_div
        
    def predict_with_uncertainty(self, 
                                state: torch.Tensor,
                                action: Optional[torch.Tensor] = None,
                                num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimates.
        
        Args:
            state: State tensor
            action: Optional action tensor
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        with torch.no_grad():
            samples = self.forward(state, action, num_samples=num_samples)
            mean_pred = torch.mean(samples, dim=0)
            uncertainty = torch.std(samples, dim=0)
            
        return mean_pred, uncertainty

class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational weights."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(weight_std)
        weight = self.weight_mu + weight_std * weight_eps
        
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(bias_std)
        bias = self.bias_mu + bias_std * bias_eps
        
        return F.linear(x, weight, bias)
        
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence between posterior and prior."""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2 + 
            weight_var / self.prior_std**2 - 
            self.weight_logvar + 
            torch.log(torch.tensor(self.prior_std**2)) - 1
        )
        
        # KL divergence for biases
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std**2 + 
            bias_var / self.prior_std**2 - 
            self.bias_logvar + 
            torch.log(torch.tensor(self.prior_std**2)) - 1
        )
        
        return weight_kl + bias_kl
