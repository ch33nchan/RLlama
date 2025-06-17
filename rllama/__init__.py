# rllama/__init__.py
"""
RLlama: A Composable Reward Engineering Framework for Reinforcement Learning.
"""

# Expose the main, user-facing classes for easy import
from .engine import RewardEngine
from .integration.trl_wrapper import TRLRlamaRewardProcessor
from .integration.sb3_wrapper import SB3RllamaRewardWrapper
from .rewards.base import BaseReward

__version__ = "0.2.1"  # Let's version this fix

__all__ = [
    "RewardEngine",
    "TRLRlamaRewardProcessor",
    "SB3RllamaRewardWrapper",
    "BaseReward",
]