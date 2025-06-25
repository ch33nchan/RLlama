#!/usr/bin/env python3
"""
Reinforcement Learning from Human Feedback (RLHF) module for RLlama.
Provides comprehensive RLHF capabilities including preference collection, training, and active learning.
"""

from .preference import PreferenceDataset, PreferenceTrainer
from .collector import PreferenceCollector, ActivePreferenceCollector

__all__ = [
    "PreferenceDataset",
    "PreferenceTrainer", 
    "PreferenceCollector",
    "ActivePreferenceCollector"
]

# Version information
__version__ = "0.7.0"

# RLHF component registry for dynamic instantiation
RLHF_REGISTRY = {
    "PreferenceDataset": PreferenceDataset,
    "PreferenceTrainer": PreferenceTrainer,
    "PreferenceCollector": PreferenceCollector,
    "ActivePreferenceCollector": ActivePreferenceCollector
}

def get_rlhf_component(component_name: str):
    """
    Get an RLHF component class by name.
    
    Args:
        component_name: Name of the RLHF component
        
    Returns:
        Component class if found, None otherwise
    """
    return RLHF_REGISTRY.get(component_name)

def create_rlhf_component(component_name: str, **kwargs):
    """
    Create an RLHF component instance by name.
    
    Args:
        component_name: Name of the RLHF component
        **kwargs: Arguments to pass to component constructor
        
    Returns:
        Component instance if successful, None otherwise
    """
    component_class = get_rlhf_component(component_name)
    if component_class is None:
        return None
    
    try:
        return component_class(**kwargs)
    except Exception:
        return None

def list_rlhf_components():
    """List all available RLHF component types."""
    return list(RLHF_REGISTRY.keys())
