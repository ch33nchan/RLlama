#!/usr/bin/env python3
"""
Neural network models for RLlama reward engineering framework.
Provides base classes and implementations for reward modeling.
"""

from .base import BaseRewardModel
from .reward_models import MLPRewardModel, EnsembleRewardModel
from .trainer import RewardModelTrainer

__all__ = [
    "BaseRewardModel",
    "MLPRewardModel", 
    "EnsembleRewardModel",
    "RewardModelTrainer"
]

# Version information
__version__ = "0.7.0"

# Model registry for dynamic instantiation
MODEL_REGISTRY = {
    "MLPRewardModel": MLPRewardModel,
    "EnsembleRewardModel": EnsembleRewardModel
}

def get_model_class(model_name: str):
    """
    Get a model class by name.
    
    Args:
        model_name: Name of the model class
        
    Returns:
        Model class if found, None otherwise
    """
    return MODEL_REGISTRY.get(model_name)

def create_model(model_name: str, **kwargs):
    """
    Create a model instance by name.
    
    Args:
        model_name: Name of the model class
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        Model instance if successful, None otherwise
    """
    model_class = get_model_class(model_name)
    if model_class is None:
        return None
    
    try:
        return model_class(**kwargs)
    except Exception:
        return None

def list_models():
    """List all available model types."""
    return list(MODEL_REGISTRY.keys())
