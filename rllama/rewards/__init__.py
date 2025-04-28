# Import core classes for easier access
from .base import BaseReward
from .composition import RewardComposer
# Remove ScheduleType from this import
from .shaping import RewardShaper, RewardConfig
from .registry import reward_registry

# Import specific reward implementations to ensure they are registered
from . import specific_rewards # Add this line

__all__ = [
    "BaseReward",
    "RewardComposer",
    "RewardShaper",
    "RewardConfig",
    # Remove ScheduleType from __all__ list
    "reward_registry",
    # Add specific reward class names here if you want them directly accessible, e.g.
    # "ToxicityPenalty",
]