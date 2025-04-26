# Existing imports
from .base import BaseReward
from .common import StepPenaltyReward, GoalReward, HolePenaltyReward # Add HolePenaltyReward
from .composition import RewardComposer
from .shaping import RewardShaper, RewardConfig
from .registry import RewardRegistry, register_reward, get_reward_component
#from .dashboard import RewardDashboard
from .optimization import BayesianRewardOptimizer

__all__ = [
    "BaseReward",
    "StepPenaltyReward",
    "GoalReward",
    "HolePenaltyReward", # Add to __all__
    "RewardComposer",
    "RewardShaper",
    "RewardConfig",
    "RewardRegistry",
    "register_reward",
    "get_reward_component",
    "RewardDashboard",
    "BayesianRewardOptimizer",
]