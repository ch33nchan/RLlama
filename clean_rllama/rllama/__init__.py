"""
RLlama: A composable reward engineering framework for reinforcement learning
"""

__version__ = "0.1.0"

# Import core components for convenience
from .engine import RewardEngine
from .rewards.base import BaseReward

__all__ = [
    "RewardEngine",
    "BaseReward",
]
