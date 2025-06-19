# RLlama

<p align="center">
  <img src="docs/images/llamagym.jpg" alt="RLlama Logo" width="200"/>
</p>

<p align="center">
    <em>A composable reward engineering framework for reinforcement learning.</em>
</p>

---

## Features

- 🧩 **Modular Reward Components**: Mix and match reward functions to shape agent behavior
- 🔍 **Reward Optimization**: Automatically tune reward weights with Bayesian optimization
- 🧠 **Memory Systems**: Episodic and working memory for improved agent capabilities
- 📊 **Visualization Tools**: Track and analyze reward contributions
- 🔗 **RL Library Integration**: Seamless integration with OpenAI Gym and Stable Baselines3
- 💬 **RLHF Support**: Tools for Reinforcement Learning from Human Feedback
- 🌐 **Neural Network Reward Models**: Deep learning based reward modeling
- 🎛️ **Reward Normalization**: Multiple strategies for normalizing rewards

## Installation

```bash
# Install from PyPI
pip install rllama

# Install with all optional dependencies
pip install "rllama[all]"

# Install in development mode
git clone https://github.com/ch33nchanyes/rllama.git
cd rllama
pip install -e ".[dev]"
Quick Start
Python
from rllama import RewardEngine

# Initialize the reward engine with a config file
engine = RewardEngine('path/to/config.yaml')

# Use the engine to compute rewards
context = {"response": "This is a test response"}
reward = engine.compute(context)
print(f"Computed reward: {reward}")
Configuration
RLlama uses YAML configuration files to specify reward components and their parameters:

YAML
reward_components:
  - name: LengthReward
    params:
      target_length: 100
      strength: 0.001
  - name: DiversityReward
    params:
      history_size: 10
      strength: 0.5

shaping_config:
  LengthReward:
    weight: 0.5
  DiversityReward:
    weight: 1.0

logging:
  log_dir: ./reward_logs
  log_frequency: 100
Available Reward Components
RLlama includes many built-in reward components:

Basic Components
LengthReward: Rewards based on response length
ConstantReward: Provides a fixed reward value
DiversityReward: Rewards diversity in actions or outputs
Advanced Components
CuriosityReward: Rewards exploration of novel states
ProgressReward: Rewards progress toward a goal
AdaptiveClippingReward: Adaptively clips rewards
GradualCurriculumReward: Increases difficulty progressively
MetaLearningReward: Adapts to agent's learning progress
HindsightExperienceReward: Implements hindsight experience replay
Neural Network Reward Models
RLlama supports neural network-based reward models:

Python
from rllama.models import MLPRewardModel, RewardModelTrainer
import torch

# Create a reward model
model = MLPRewardModel(input_dim=8, hidden_dims=[64, 32])

# Train the model
trainer = RewardModelTrainer(model=model)
trainer.train(train_loader, val_loader, epochs=50)
Reinforcement Learning from Human Feedback (RLHF)
Learn reward models from human preferences:

Python
from rllama.rlhf import PreferenceTrainer, PreferenceDataset
from rllama.models import MLPRewardModel

# Create preference dataset
dataset = PreferenceDataset(states_a, states_b, preferences)

# Train model from preferences
model = MLPRewardModel(input_dim=8)
trainer = PreferenceTrainer(model=model)
trainer.train(train_loader, val_loader, epochs=50)
Documentation
For detailed documentation, please see:

Concepts: Core concepts and architecture
Usage Guide: Detailed usage instructions
Reward Cookbook: Recipes for common reward functions
Optimization Guide: Guide to reward optimization
License
MIT License

Copyright (c) 2025 Srinivas
