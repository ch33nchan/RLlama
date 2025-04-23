from typing import Dict, Type, Any
from .base import BaseReward
from .common import GoalReward, StepPenaltyReward, ActionNoveltyReward
# Import specific components if they are intended for general use
# Or handle them via custom registration if they are example-specific
# from examples.reward_integration_demo import FrozenLakeGoalReward, FrozenLakeHolePenalty # Avoid this tight coupling

# Define a type hint for reward component classes
RewardComponentClass = Type[BaseReward]

REWARD_REGISTRY: Dict[str, RewardComponentClass] = {
    "goal": GoalReward,
    "step_penalty": StepPenaltyReward,
    "action_novelty": ActionNoveltyReward,
    # We will define FrozenLake specific ones in the demo script for now,
    # or you could move them to common.py if they are general enough.
}

# Optional: Function to register custom components dynamically if needed
def register_reward_component(name: str, component_class: RewardComponentClass):
    """Registers a custom reward component class."""
    if name in REWARD_REGISTRY:
        print(f"Warning: Overwriting existing reward component in registry: {name}")
    REWARD_REGISTRY[name] = component_class

def create_reward_component(name: str, **kwargs: Any) -> BaseReward:
    """
    Instantiates a reward component from the registry using its name.
    Passes keyword arguments to the component's __init__ method.
    """
    if name not in REWARD_REGISTRY:
        # Consider searching in imported modules or raising a clearer error
        raise ValueError(f"Unknown reward component name: '{name}'. Available: {list(REWARD_REGISTRY.keys())}")

    component_class = REWARD_REGISTRY[name]
    try:
        # Instantiate the class with provided kwargs
        return component_class(**kwargs)
    except TypeError as e:
        raise TypeError(f"Error initializing component '{name}' with args {kwargs}: {e}") from e