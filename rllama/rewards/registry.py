# rllama/rewards/registry.py

from typing import Dict, Any, Callable, Type, Optional
import inspect

# Registry to store reward components
REWARD_REGISTRY = {}

def register_reward_component(cls: Type) -> Type:
    """
    Decorator to register a reward component class.
    
    Args:
        cls: The reward component class to register
        
    Returns:
        The registered class (unchanged)
    """
    component_name = cls.__name__
    REWARD_REGISTRY[component_name] = cls
    return cls

def get_reward_component(name: str) -> Optional[Type]:
    """
    Get a reward component class by name.
    
    Args:
        name: The name of the reward component
        
    Returns:
        The reward component class, or None if not found
    """
    return REWARD_REGISTRY.get(name)
