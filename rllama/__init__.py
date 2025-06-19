"""
RLlama: A composable reward engineering framework for reinforcement learning
"""

__version__ = "0.7.0"

# Import core components for convenience
from .engine import RewardEngine
from .rewards.base import BaseReward
from .rewards.registry import register_reward_component
from .memory import MemoryEntry, EpisodicMemory, WorkingMemory, MemoryCompressor

# Import reward components
from .rewards.components import (
    LengthReward, 
    ConstantReward, 
    DiversityReward, 
    CuriosityReward, 
    ProgressReward,
    # Advanced learning components
    AdaptiveClippingReward,
    GradualCurriculumReward,
    UncertaintyBasedReward,
    MetaLearningReward,
    HindsightExperienceReward,
    AdversarialReward,
    RobustnessReward
)

# Import models
from .models import (
    BaseRewardModel,
    MLPRewardModel,
    EnsembleRewardModel
)

# Import RLHF components
from .rlhf import (
    PreferenceDataset,
    PreferenceTrainer,
    PreferenceCollector,
    ActivePreferenceCollector
)

__all__ = [
    # Core
    "RewardEngine",
    "BaseReward",
    "register_reward_component",
    
    # Memory
    "MemoryEntry", 
    "EpisodicMemory", 
    "WorkingMemory", 
    "MemoryCompressor",
    
    # Basic rewards
    "LengthReward",
    "ConstantReward",
    "DiversityReward", 
    "CuriosityReward", 
    "ProgressReward",
    
    # Advanced learning rewards
    "AdaptiveClippingReward",
    "GradualCurriculumReward",
    "UncertaintyBasedReward",
    "MetaLearningReward",
    "HindsightExperienceReward",
    "AdversarialReward",
    "RobustnessReward",
    
    # Models
    "BaseRewardModel",
    "MLPRewardModel",
    "EnsembleRewardModel",
    
    # RLHF
    "PreferenceDataset",
    "PreferenceTrainer",
    "PreferenceCollector",
    "ActivePreferenceCollector"
]
