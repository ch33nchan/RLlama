from typing import Type, Dict, Any, Optional, Callable # Added Callable
import logging
from .base import BaseReward

logger = logging.getLogger(__name__)

class RewardRegistry:
    """Singleton registry for reward components."""
    _instance = None
    _registry: Dict[str, Type[BaseReward]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RewardRegistry, cls).__new__(cls)
            # Initialize registry dictionary only once
            cls._registry = {}
            # --- Auto-register common components ---
            # We will call register using the decorator pattern now if needed,
            # or keep the direct calls if preferred for common ones.
            # Let's keep direct calls here for simplicity unless common.py also uses decorators.
            cls._instance._register_common()
            # --------------------------------------
        return cls._instance

    def _register_common(self):
        """Registers standard components included with the library."""
        try:
            # Import common components here to avoid circular dependencies at module level
            from .common import StepPenaltyReward, GoalReward, HolePenaltyReward

            # Use the direct registration method internally if preferred
            self._register_directly("step_penalty", StepPenaltyReward)
            self._register_directly("goal_reward", GoalReward)
            self._register_directly("hole_penalty", HolePenaltyReward)
            logger.info("Auto-registered common reward components: step_penalty, goal_reward, hole_penalty")
        except ImportError as e:
            logger.warning(f"Could not auto-register common rewards (optional): {e}")
        except TypeError as e:
             logger.error(f"TypeError during auto-registration of common rewards: {e}")


    # Internal method for direct registration
    def _register_directly(self, name: str, component_class: Type[BaseReward]):
        if not issubclass(component_class, BaseReward):
            raise TypeError(f"Class {component_class.__name__} must inherit from BaseReward.")
        if name in self._registry:
            logger.warning(f"Reward component '{name}' is already registered. Overwriting.")
        self._registry[name] = component_class
        logger.debug(f"Registered reward component directly: '{name}' -> {component_class.__name__}")

    # Modified register method to support decorator usage
    def register(self, name: str) -> Callable[[Type[BaseReward]], Type[BaseReward]]:
        """
        Registers a reward component class under a given name.
        Can be used as a decorator factory: @reward_registry.register("my_reward")
        """
        def decorator(component_class: Type[BaseReward]) -> Type[BaseReward]:
            if not issubclass(component_class, BaseReward):
                # Raise error early if the class type is wrong
                raise TypeError(f"Class {component_class.__name__} must inherit from BaseReward to be registered.")
            if name in self._registry:
                logger.warning(f"Reward component '{name}' is already registered. Overwriting.")
            self._registry[name] = component_class
            logger.debug(f"Registered reward component via decorator: '{name}' -> {component_class.__name__}")
            return component_class # Return the original class

        return decorator # Return the actual decorator function

    def get_class(self, name: str) -> Optional[Type[BaseReward]]:
        """Gets the class associated with a registered name."""
        component_class = self._registry.get(name)
        # Optional: Add logging if class not found
        # if component_class is None:
        #     logger.warning(f"Reward component '{name}' not found in registry.")
        return component_class

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> BaseReward:
        """Creates an instance of a registered reward component."""
        component_class = self.get_class(name)
        if component_class is None:
            # Make the error message more informative
            raise ValueError(f"Reward component '{name}' not found in registry. Available: {list(self._registry.keys())}")
        try:
            # Ensure params is a dictionary, even if empty
            instance_params = params or {}
            # Pass name explicitly if the component's __init__ expects it
            # This depends on your BaseReward and specific component design.
            # If components always take 'name' based on the config key,
            # the config loading logic should handle passing it, not the registry 'create'.
            # However, if 'create' is used elsewhere, you might need it.
            # Let's assume config loading handles passing 'name' for now.
            instance = component_class(**instance_params)
            return instance
        except TypeError as e:
            logger.error(f"Failed to instantiate component '{name}' with params {params}. Check __init__ signature: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to instantiate component '{name}' with params {params}: {e}")
            raise

# --- Global functions for convenience ---
# Create a global instance of the registry for other modules to import
reward_registry = RewardRegistry()

# Optional: Update global functions if they should use the new registry logic or are deprecated
# These might cause issues if they reference a different instance or old methods.
# Consider removing them if `reward_registry.register` and `reward_registry.create` are used directly.

# def register_reward(name: str, component_class: Type[BaseReward]):
#     """Registers a reward component class globally."""
#     # Ensure this uses the correct instance and method
#     reward_registry._register_directly(name, component_class) # Example: Use direct registration for this function

# def get_reward_component(name: str, params: Optional[Dict[str, Any]] = None) -> BaseReward:
#     """Creates an instance of a registered reward component globally."""
#     return reward_registry.create(name, params)

# def get_reward_class(name: str) -> Optional[Type[BaseReward]]:
#     """Gets the class associated with a registered name globally."""
#     return reward_registry.get_class(name)