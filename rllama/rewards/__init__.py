# Import core classes for easier access
from .base import BaseReward
from .composition import RewardComposer
# Remove ScheduleType from this import
from .shaping import RewardShaper, RewardConfig
from .registry import reward_registry
# from .dashboard import RewardDashboard # <<< Comment out this line
from .registry import register_reward_component, get_reward_component_class, get_reward_component
from .optimizer import BayesianRewardOptimizer

# Optionally, import common components to make them easily accessible
# from .components.common import StepPenaltyReward, ...

__all__ = [
    "BaseReward",
    "RewardComposer",
    "RewardShaper",
    "RewardConfig",
    # Remove ScheduleType from __all__ list
    "reward_registry",
    "register_reward_component", # <<< Ensure these were added correctly
    "get_reward_component_class",# <<< Ensure these were added correctly
    "get_reward_component",      # <<< Ensure these were added correctly
    "BayesianRewardOptimizer",   # <<< Ensure these were added correctly
    # "RewardDashboard", # <<< Comment out this line too if it exists in your __all__ list
    # Add specific reward class names here if you want them directly accessible, e.g.
    # "ToxicityPenalty",
]