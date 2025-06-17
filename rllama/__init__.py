"""
RLlama: Composable Reward Engineering Framework for Reinforcement Learning
"""

__version__ = "0.1.1"
__author__ = "RLlama Team"

# Core imports
from .core.composer import RewardComposer
from .core.shaper import RewardShaper

# Integration imports
try:
    from .integration.trl_wrapper import TRLRllamaRewardProcessor, TRLRllamaCallback
except ImportError:
    # TRL not installed
    pass

__all__ = [
    "RewardComposer",
    "RewardShaper",
    "TRLRllamaRewardProcessor", 
    "TRLRllamaCallback"
]