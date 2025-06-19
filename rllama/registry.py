# rllama/rewards/registry.py

import inspect
from typing import Dict, Type, List, Any, Optional, Union, Callable
from .base import BaseReward

class RewardRegistry:
    """
    Registry for reward components that allows dynamic registration and lookup.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type[BaseReward]] = {}
    
    def register(self, reward_class_or_name: Union[Type[BaseReward], str] = None) -> Callable:
        """
        Register a reward component class.
        Can be used as a decorator or a function.
        
        Args:
            reward_class_or_name: Reward class to register, or name to use for registration.
            
        Returns:
            Decorator function that registers the class.
        """
        def decorator(cls):
            # If called as @register
            if reward_class_or_name is None:
                name = cls.__name__
            # If called as @register("name")
            elif isinstance(reward_class_or_name, str):
                name = reward_class_or_name
            # If called as register(cls)
            else:
                name = reward_class_or_name.__name__
                cls = reward_class_or_name
                
            # Check if class inherits from BaseReward
            if not issubclass(cls, BaseReward):
                raise TypeError(f"Reward class {name} must inherit from BaseReward")
                
            # Register the class
            self._registry[name] = cls
            return cls
            
        # Handle case where decorator is called directly with a class
        if isinstance(reward_class_or_name, type) and issubclass(reward_class_or_name, BaseReward):
            return decorator(reward_class_or_name)
            
        return decorator
    
    def get(self, name: str) -> Optional[Type[BaseReward]]:
        """
        Get a reward component class by name.
        
        Args:
            name: Name of the reward component.
            
        Returns:
            The reward component class if found, None otherwise.
        """
        return self._registry.get(name)
    
    def create(self, 
               name: str, 
               **kwargs) -> Optional[BaseReward]:
        """
        Create an instance of a reward component.
        
        Args:
            name: Name of the reward component.
            **kwargs: Arguments to pass to the constructor.
            
        Returns:
            An instance of the reward component if found, None otherwise.
        """
        reward_class = self.get(name)
        if reward_class is None:
            return None
            
        # Filter kwargs to only include parameters accepted by the constructor
        valid_params = inspect.signature(reward_class.__init__).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        return reward_class(**filtered_kwargs)
    
    def list_rewards(self) -> List[str]:
        """
        List all registered reward components.
        
        Returns:
            List of reward component names.
        """
        return list(self._registry.keys())
    
    def describe_reward(self, name: str) -> Dict[str, Any]:
        """
        Get information about a reward component.
        
        Args:
            name: Name of the reward component.
            
        Returns:
            Dictionary with information about the component.
        """
        reward_class = self.get(name)
        if reward_class is None:
            return {"error": f"Reward component {name} not found"}
            
        # Get docstring
        docstring = inspect.getdoc(reward_class) or ""
        
        # Get constructor parameters
        signature = inspect.signature(reward_class.__init__)
        parameters = {}
        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue
                
            parameters[param_name] = {
                "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
                "annotation": str(param.annotation) if param.annotation is not inspect.Parameter.empty else None,
                "required": param.default is inspect.Parameter.empty and param_name != "kwargs"
            }
        
        return {
            "name": name,
            "description": docstring,
            "parameters": parameters,
            "module": reward_class.__module__
        }


# Create global registry instance
reward_registry = RewardRegistry()

# Function aliases for backward compatibility
def register_reward_component(reward_class_or_name=None):
    """Alias for reward_registry.register"""
    return reward_registry.register(reward_class_or_name)
    
def get_reward_component_class(name: str) -> Optional[Type[BaseReward]]:
    """Alias for reward_registry.get"""
    return reward_registry.get(name)
    
def get_reward_component(name: str, **kwargs) -> Optional[BaseReward]:
    """Alias for reward_registry.create"""
    return reward_registry.create(name, **kwargs)


# Import and register standard reward components
from .components.common import LengthReward, ConstantReward

reward_registry.register(LengthReward)
reward_registry.register(ConstantReward)

# Create a legacy REWARD_REGISTRY dict for backward compatibility
REWARD_REGISTRY = {
    "LengthReward": LengthReward,
    "ConstantReward": ConstantReward,
}