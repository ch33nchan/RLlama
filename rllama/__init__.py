#!/usr/bin/env python3
"""
RLlama: A composable reward engineering framework for reinforcement learning.

RLlama provides a comprehensive framework for designing, implementing, and optimizing
reward functions in reinforcement learning systems. It supports:

- Modular reward component design
- Advanced reward composition and shaping
- Neural network reward models
- Reinforcement Learning from Human Feedback (RLHF)
- Sophisticated memory systems
- Bayesian optimization for hyperparameters
- Comprehensive logging and analytics

Example usage:
    >>> from rllama import RewardEngine
    >>> engine = RewardEngine("config.yaml")
    >>> reward = engine.compute({"response": "Hello world!", "query": "Say hello"})
"""

__version__ = "0.7.0"
__author__ = "RLlama Team"
__email__ = "contact@rllama.ai"
__license__ = "MIT"

# Core framework components
from .engine import RewardEngine
from .logger import RewardLogger

# Base classes and registry
from .rewards.base import BaseReward
from .rewards.registry import (
    reward_registry, 
    register_reward_component, 
    get_reward_component_class,
    get_reward_component,
    REWARD_REGISTRY  # Legacy compatibility
)

# Reward composition and shaping
from .rewards.composer import RewardComposer
from .rewards.shaper import RewardShaper, RewardConfig, ScheduleType

# Memory systems
from .memory import (
    MemoryEntry, 
    EpisodicMemory, 
    WorkingMemory, 
    MemoryCompressor
)

# Basic reward components
from .rewards.components.common import (
    LengthReward, 
    ConstantReward,
    ThresholdReward,
    RangeReward,
    ProportionalReward,
    BinaryReward
)

# Length-specific rewards
from .rewards.components.length_rewards import (
    TokenLengthReward,
    SentenceLengthReward,
    ParagraphLengthReward,
    OptimalLengthReward
)

# Specific domain rewards
from .rewards.components.specific_rewards import (
    DiversityReward, 
    CuriosityReward, 
    ProgressReward,
    CoherenceReward,
    RelevanceReward,
    SafetyReward
)

# Advanced reward components
from .rewards.components.advanced import (
    TemporalConsistencyReward,
    MultiObjectiveReward,
    HierarchicalReward,
    ContrastiveReward
)

# Learning-based components
from .rewards.components.learning.adaptive import (
    AdaptiveClippingReward,
    GradualCurriculumReward,
    AdaptiveNormalizationReward
)

from .rewards.components.learning.adversarial import (
    AdversarialReward,
    RobustnessReward
)

from .rewards.components.learning.meta import (
    MetaLearningReward,
    UncertaintyBasedReward,
    HindsightExperienceReward
)

# LLM-specific components
from .rewards.components.llm_rewards import (
    PerplexityReward,
    SemanticSimilarityReward,
    ToxicityReward,
    FactualityReward,
    CreativityReward
)

# Robotics components
from .rewards.components.robotics_components import (
    CollisionAvoidanceReward,
    EnergyEfficiencyReward,
    TaskCompletionReward,
    SmoothTrajectoryReward
)

# Neural network models
from .models.base import BaseRewardModel
from .models.reward_models import (
    MLPRewardModel,
    EnsembleRewardModel,
    BayesianRewardModel
)
from .models.trainer import RewardModelTrainer

# RLHF components
from .rlhf.preference import PreferenceDataset, PreferenceTrainer
from .rlhf.collector import PreferenceCollector, ActivePreferenceCollector

# Optimization
from .rewards.optimizer import (
    BayesianRewardOptimizer,
    GaussianProcessOptimizer,
    MultiObjectiveOptimizer,
    OptimizationResults,
    create_optimizer
)

# Define public API
__all__ = [
    # Core framework
    "RewardEngine",
    "RewardLogger",
    
    # Base classes and registry
    "BaseReward",
    "reward_registry",
    "register_reward_component",
    "get_reward_component_class", 
    "get_reward_component",
    "REWARD_REGISTRY",
    
    # Composition and shaping
    "RewardComposer",
    "RewardShaper",
    "RewardConfig", 
    "ScheduleType",
    
    # Memory systems
    "MemoryEntry", 
    "EpisodicMemory", 
    "WorkingMemory", 
    "MemoryCompressor",
    
    # Basic reward components
    "LengthReward",
    "ConstantReward",
    "ThresholdReward",
    "RangeReward", 
    "ProportionalReward",
    "BinaryReward",
    
    # Length-specific rewards
    "TokenLengthReward",
    "SentenceLengthReward",
    "ParagraphLengthReward", 
    "OptimalLengthReward",
    
    # Specific domain rewards
    "DiversityReward", 
    "CuriosityReward", 
    "ProgressReward",
    "CoherenceReward",
    "RelevanceReward",
    "SafetyReward",
    
    # Advanced rewards
    "TemporalConsistencyReward",
    "MultiObjectiveReward",
    "HierarchicalReward",
    "ContrastiveReward",
    
    # Learning-based rewards
    "AdaptiveClippingReward",
    "GradualCurriculumReward", 
    "AdaptiveNormalizationReward",
    "AdversarialReward",
    "RobustnessReward",
    "MetaLearningReward",
    "UncertaintyBasedReward",
    "HindsightExperienceReward",
    
    # LLM-specific rewards
    "PerplexityReward",
    "SemanticSimilarityReward",
    "ToxicityReward",
    "FactualityReward", 
    "CreativityReward",
    
    # Robotics rewards
    "CollisionAvoidanceReward",
    "EnergyEfficiencyReward",
    "TaskCompletionReward",
    "SmoothTrajectoryReward",
    
    # Neural network models
    "BaseRewardModel",
    "MLPRewardModel",
    "EnsembleRewardModel",
    "BayesianRewardModel",
    "RewardModelTrainer",
    
    # RLHF
    "PreferenceDataset",
    "PreferenceTrainer",
    "PreferenceCollector",
    "ActivePreferenceCollector",
    
    # Optimization
    "BayesianRewardOptimizer",
    "GaussianProcessOptimizer", 
    "MultiObjectiveOptimizer",
    "OptimizationResults",
    "create_optimizer"
]

# Framework metadata
FRAMEWORK_INFO = {
    "name": "RLlama",
    "version": __version__,
    "description": "A composable reward engineering framework for reinforcement learning",
    "components": {
        "basic_rewards": 6,
        "length_rewards": 4, 
        "specific_rewards": 6,
        "advanced_rewards": 4,
        "learning_rewards": 8,
        "llm_rewards": 5,
        "robotics_rewards": 4,
        "total_components": 37
    },
    "features": [
        "Modular reward design",
        "Advanced composition strategies", 
        "Neural network reward models",
        "RLHF support",
        "Memory systems",
        "Bayesian optimization",
        "Comprehensive logging"
    ]
}

def get_framework_info() -> dict:
    """Get comprehensive information about the RLlama framework."""
    return FRAMEWORK_INFO.copy()

def list_all_components() -> dict:
    """List all available reward components by category."""
    return {
        "basic": [
            "LengthReward", "ConstantReward", "ThresholdReward", 
            "RangeReward", "ProportionalReward", "BinaryReward"
        ],
        "length": [
            "TokenLengthReward", "SentenceLengthReward", 
            "ParagraphLengthReward", "OptimalLengthReward"
        ],
        "specific": [
            "DiversityReward", "CuriosityReward", "ProgressReward",
            "CoherenceReward", "RelevanceReward", "SafetyReward"
        ],
        "advanced": [
            "TemporalConsistencyReward", "MultiObjectiveReward",
            "HierarchicalReward", "ContrastiveReward"
        ],
        "learning": [
            "AdaptiveClippingReward", "GradualCurriculumReward",
            "AdaptiveNormalizationReward", "AdversarialReward", 
            "RobustnessReward", "MetaLearningReward",
            "UncertaintyBasedReward", "HindsightExperienceReward"
        ],
        "llm": [
            "PerplexityReward", "SemanticSimilarityReward",
            "ToxicityReward", "FactualityReward", "CreativityReward"
        ],
        "robotics": [
            "CollisionAvoidanceReward", "EnergyEfficiencyReward",
            "TaskCompletionReward", "SmoothTrajectoryReward"
        ]
    }

def create_reward_engine(config_path: str, **kwargs) -> RewardEngine:
    """
    Convenience function to create a RewardEngine instance.
    
    Args:
        config_path: Path to the YAML configuration file
        **kwargs: Additional arguments for RewardEngine
        
    Returns:
        Configured RewardEngine instance
    """
    return RewardEngine(config_path, **kwargs)

def quick_start_example():
    """
    Print a quick start example for new users.
    """
    example = """
    RLlama Quick Start Example:
    
    1. Create a configuration file (config.yaml):
    
    reward_components:
      - name: LengthReward
        params:
          target_length: 100
          strength: 0.01
      - name: DiversityReward
        params:
          history_size: 10
          strength: 0.5
    
    shaping_config:
      LengthReward: 1.0
      DiversityReward: 0.5
    
    2. Use the framework:
    
    from rllama import RewardEngine
    
    # Initialize engine
    engine = RewardEngine("config.yaml", verbose=True)
    
    # Compute reward
    context = {
        "response": "This is a sample response text.",
        "query": "Generate a response"
    }
    reward = engine.compute_and_log(context)
    print(f"Computed reward: {reward}")
    
    For more examples, see the examples/ directory.
    """
    print(example)

# Version compatibility check
def check_dependencies():
    """Check if optional dependencies are available."""
    dependencies = {
        "torch": False,
        "numpy": False, 
        "optuna": False,
        "sklearn": False,
        "transformers": False
    }
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        pass
        
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        pass
        
    try:
        import optuna
        dependencies["optuna"] = True
    except ImportError:
        pass
        
    try:
        import sklearn
        dependencies["sklearn"] = True
    except ImportError:
        pass
        
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
        
    return dependencies

# Initialize framework
def _initialize_framework():
    """Initialize the RLlama framework."""
    # Auto-register all components
    reward_registry._auto_discovery_enabled = True
    
    # Set up logging
    import logging
    logging.getLogger("rllama").setLevel(logging.INFO)

# Run initialization
_initialize_framework()

# Framework banner
BANNER = f"""
╔══════════════════════════════════════════════════════════════╗
║                           RLlama v{__version__}                           ║
║        A Composable Reward Engineering Framework             ║
║                                                              ║
║  🦙 {FRAMEWORK_INFO['components']['total_components']} reward components available                        ║
║  🧠 Neural network reward models                            ║
║  🤖 RLHF support                                            ║
║  📊 Advanced optimization                                    ║
║  💾 Memory systems                                          ║
║                                                              ║
║  Get started: rllama.quick_start_example()                  ║
╚══════════════════════════════════════════════════════════════╝
"""

def show_banner():
    """Display the RLlama framework banner."""
    print(BANNER)

# Auto-show banner on import (can be disabled)
import os
if os.getenv("RLLAMA_SHOW_BANNER", "1") == "1":
    show_banner()
