from .base import BaseReward
from .common import GoalReward, StepPenaltyReward, ActionNoveltyReward
# Import your custom ones if they become common
# from .custom import FrozenLakeGoalReward, FrozenLakeHolePenalty

REWARD_REGISTRY = {
    "goal": GoalReward,
    "step_penalty": StepPenaltyReward,
    "action_novelty": ActionNoveltyReward,
    # Add custom ones if desired
    # "frozen_lake_goal": FrozenLakeGoalReward,
    # "frozen_lake_hole": FrozenLakeHolePenalty,
}

def create_reward_component(name: str, **kwargs) -> BaseReward:
    """Instantiates a reward component from the registry."""
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward component name: {name}")
    component_class = REWARD_REGISTRY[name]
    # Pass kwargs to the component's __init__ (e.g., penalty value)
    return component_class(**kwargs)