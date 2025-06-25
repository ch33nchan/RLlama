"""
RLlama Reward Components

This module contains all reward components for the RLlama framework.
Components are organized into different categories for better maintainability.
"""

# Basic reward components
from .common import LengthReward, ConstantReward
from .length_rewards import (
    TokenLengthReward, 
    SentenceLengthReward, 
    ParagraphLengthReward,
    OptimalLengthReward
)
from .specific_rewards import (
    DiversityReward,
    CuriosityReward, 
    ProgressReward,
    CoherenceReward,
    RelevanceReward,
    SafetyReward
)

# Advanced components
from .advanced import (
    TemporalConsistencyReward,
    MultiObjectiveReward,
    HierarchicalReward,
    ContrastiveReward
)

# Learning-based components  
from .learning.adaptive import AdaptiveClippingReward, GradualCurriculumReward
from .learning.adversarial import AdversarialReward, RobustnessReward
from .learning.meta import (
    MetaLearningReward,
    UncertaintyBasedReward,
    HindsightExperienceReward
)

# LLM-specific components
from .llm_rewards import (
    PerplexityReward,
    SemanticSimilarityReward,
    ToxicityReward,
    FactualityReward,
    CreativityReward
)

# Robotics components
from .robotics_components import (
    CollisionAvoidanceReward,
    EnergyEfficiencyReward,
    TaskCompletionReward,
    SmoothTrajectoryReward
)

__all__ = [
    # Basic
    "LengthReward",
    "ConstantReward",
    
    # Length variants
    "TokenLengthReward",
    "SentenceLengthReward", 
    "ParagraphLengthReward",
    "OptimalLengthReward",
    
    # Specific rewards
    "DiversityReward",
    "CuriosityReward",
    "ProgressReward", 
    "CoherenceReward",
    "RelevanceReward",
    "SafetyReward",
    
    # Advanced
    "TemporalConsistencyReward",
    "MultiObjectiveReward",
    "HierarchicalReward",
    "ContrastiveReward",
    
    # Learning-based
    "AdaptiveClippingReward",
    "GradualCurriculumReward",
    "AdversarialReward",
    "RobustnessReward",
    "MetaLearningReward",
    "UncertaintyBasedReward", 
    "HindsightExperienceReward",
    
    # LLM-specific
    "PerplexityReward",
    "SemanticSimilarityReward",
    "ToxicityReward",
    "FactualityReward",
    "CreativityReward",
    
    # Robotics
    "CollisionAvoidanceReward",
    "EnergyEfficiencyReward", 
    "TaskCompletionReward",
    "SmoothTrajectoryReward"
]
