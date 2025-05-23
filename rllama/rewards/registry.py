from typing import Type, Dict, Any, Optional, Callable, List # Added List
import logging
from .base import BaseReward

logger = logging.getLogger(__name__)

class RewardRegistry:
    """Singleton registry for reward components."""
    _instance: Optional['RewardRegistry'] = None
    _registry: Dict[str, Type[BaseReward]]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RewardRegistry, cls).__new__(cls)
            # Initialize registry dictionary only once when the instance is created
            cls._instance._registry = {} 
            cls._instance._register_common()
        return cls._instance

    def _register_common(self):
        """Registers standard components included with the library."""
        try:
            from .common import StepPenaltyReward, GoalReward, HolePenaltyReward # Assuming these exist

            self._register_directly("step_penalty", StepPenaltyReward)
            self._register_directly("goal_reward", GoalReward)
            self._register_directly("hole_penalty", HolePenaltyReward)
            logger.info("Auto-registered common reward components: step_penalty, goal_reward, hole_penalty")
        except ImportError as e:
            logger.warning(f"Could not auto-register common rewards (optional, some may not exist): {e}")
        except TypeError as e:
             logger.error(f"TypeError during auto-registration of common rewards: {e}")
        except Exception as e: # Catch any other exception during common registration
            logger.error(f"Unexpected error during auto-registration of common rewards: {e}")


    def _register_directly(self, name: str, component_class: Type[BaseReward]):
        """Internal method for direct registration."""
        if not isinstance(component_class, type) or not issubclass(component_class, BaseReward):
            raise TypeError(f"Class {component_class.__name__} must be a class and inherit from BaseReward.")
        if name in self._registry:
            if self._registry[name] == component_class:
                logger.debug(f"Reward component '{name}' with class {component_class.__name__} is already registered. Skipping.")
                return
            logger.warning(f"Reward component '{name}' is already registered with {self._registry[name].__name__}. Overwriting with {component_class.__name__}.")
        self._registry[name] = component_class
        logger.debug(f"Registered reward component directly: '{name}' -> {component_class.__name__}")

    def register(self, name: str, component_class: Optional[Type[BaseReward]] = None) -> Callable[[Type[BaseReward]], Type[BaseReward]]:
        """
        Registers a reward component class under a given name.
        Can be used as a direct call: reward_registry.register("my_reward", MyRewardClass)
        Or as a decorator factory: @reward_registry.register("my_reward")
        """
        def decorator(cls_to_register: Type[BaseReward]) -> Type[BaseReward]:
            if not isinstance(cls_to_register, type) or not issubclass(cls_to_register, BaseReward):
                raise TypeError(f"Class {cls_to_register.__name__} must be a class and inherit from BaseReward to be registered.")
            
            if name in self._registry:
                if self._registry[name] == cls_to_register:
                    logger.debug(f"Reward component '{name}' with class {cls_to_register.__name__} is already registered. Skipping decorator registration.")
                    return cls_to_register 
                logger.warning(f"Reward component '{name}' is already registered with {self._registry[name].__name__} via decorator. Overwriting with {cls_to_register.__name__}.")
            
            self._registry[name] = cls_to_register
            logger.debug(f"Registered reward component via decorator: '{name}' -> {cls_to_register.__name__}")
            return cls_to_register

        if component_class is not None: # Direct call
            decorator(component_class) # Apply the registration logic directly
            # For direct call, the return value of register itself isn't used to decorate,
            # so we return a no-op decorator or handle it appropriately.
            # Here, we just perform the registration. The return type hint is for the decorator case.
            # To satisfy the return type for direct call, we can return a simple lambda.
            return lambda x: x # No-op decorator for direct call case
        
        return decorator # Return the actual decorator function for decorator usage

    def get_class(self, name: str) -> Optional[Type[BaseReward]]:
        """Gets the class associated with a registered name."""
        component_class = self._registry.get(name)
        if component_class is None:
            logger.warning(f"Reward component '{name}' not found in registry. Available: {list(self._registry.keys())}")
        return component_class

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> BaseReward:
        """Creates an instance of a registered reward component."""
        component_class = self.get_class(name)
        if component_class is None:
            raise ValueError(f"Reward component '{name}' not found in registry. Available: {list(self._registry.keys())}")
        
        instance_params = params or {}
        try:
            # If 'name' is a common parameter for BaseReward or its children,
            # and it's not in params, we can add it.
            # However, this assumes components might want their registration name.
            # It's often better if the config itself provides the name if needed by init.
            # if 'name' not in instance_params and hasattr(component_class, '__init__') and \
            #    'name' in component_class.__init__.__code__.co_varnames:
            #    instance_params_with_name = {'name': name, **instance_params}
            #    instance = component_class(**instance_params_with_name)
            # else:
            instance = component_class(**instance_params)
            return instance
        except TypeError as e:
            logger.error(f"Failed to instantiate component '{name}' with params {instance_params}. Check __init__ signature of {component_class.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error instantiating component '{name}' with params {instance_params}: {e}")
            raise

    def is_registered(self, name: str) -> bool:
        """Checks if a reward component name is already registered."""
        return name in self._registry

    def get_all_registered_names(self) -> List[str]:
        """Returns a list of all registered component names."""
        return list(self._registry.keys())

# Global instance of the registry
reward_registry = RewardRegistry()

# --- Global convenience functions (optional, but can be useful) ---

def register_reward_component(name: str, component_class: Type[BaseReward]):
    """Registers a reward component class globally using the reward_registry instance."""
    reward_registry.register(name, component_class) # Uses the direct call mode of register

def get_reward_component(name: str, params: Optional[Dict[str, Any]] = None) -> BaseReward:
    """Creates an instance of a registered reward component globally."""
    return reward_registry.create(name, params)

def get_reward_component_class(name: str) -> Optional[Type[BaseReward]]:
    """Gets the class associated with a registered name globally."""
    return reward_registry.get_class(name)

def is_reward_component_registered(name: str) -> bool:
    """Checks if a reward component is registered globally."""
    return reward_registry.is_registered(name)

def get_all_registered_reward_component_names() -> List[str]:
    """Gets all registered reward component names globally."""
    return reward_registry.get_all_registered_names()
