from typing import Type, Dict, Any, Optional
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
            cls._instance._register_common()
            # --------------------------------------
        return cls._instance

    def _register_common(self):
        """Registers standard components included with the library."""
        try:
            # Import common components here to avoid circular dependencies at module level
            from .common import StepPenaltyReward, GoalReward, HolePenaltyReward

            self.register("step_penalty", StepPenaltyReward)
            self.register("goal_reward", GoalReward)
            self.register("hole_penalty", HolePenaltyReward)
            logger.info("Auto-registered common reward components: step_penalty, goal_reward, hole_penalty")
        except ImportError as e:
            logger.error(f"Could not auto-register common rewards: {e}")


    def register(self, name: str, component_class: Type[BaseReward]):
        """Registers a reward component class with a given name."""
        if not issubclass(component_class, BaseReward):
            raise TypeError(f"Class {component_class.__name__} must inherit from BaseReward.")
        if name in self._registry:
            logger.warning(f"Reward component '{name}' is already registered. Overwriting.")
        self._registry[name] = component_class
        logger.debug(f"Registered reward component: '{name}' -> {component_class.__name__}")

    def get_class(self, name: str) -> Optional[Type[BaseReward]]:
        """Gets the class associated with a registered name."""
        return self._registry.get(name)

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> BaseReward:
        """Creates an instance of a registered reward component."""
        component_class = self.get_class(name)
        if component_class is None:
            raise ValueError(f"Reward component '{name}' not found in registry.")
        try:
            instance = component_class(**(params or {}))
            return instance
        except Exception as e:
            logger.error(f"Failed to instantiate component '{name}' with params {params}: {e}")
            raise

# --- Global functions for convenience ---
_registry_instance = RewardRegistry()

def register_reward(name: str, component_class: Type[BaseReward]):
    """Registers a reward component class globally."""
    _registry_instance.register(name, component_class)

def get_reward_component(name: str, params: Optional[Dict[str, Any]] = None) -> BaseReward:
    """Creates an instance of a registered reward component globally."""
    return _registry_instance.create(name, params)

def get_reward_class(name: str) -> Optional[Type[BaseReward]]:
    """Gets the class associated with a registered name globally."""
    return _registry_instance.get_class(name)