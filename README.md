# 🦙 RLlama

[![PyPI version](https://badge.fury.io/py/rllama.svg)](https://pypi.org/project/rllama/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/ch33nchan/RLlama.svg)](https://github.com/ch33nchan/RLlama/stargazers)

A **composable reward engineering framework** for reinforcement learning that makes designing, implementing, and optimizing reward functions simple and powerful.

## 🚀 Quick Start

pip install rllama

text
undefined
from rllama import RewardEngine

Initialize with configuration
engine = RewardEngine("config.yaml", verbose=True)

Compute rewards
context = {
"response": "Hello, world!",
"query": "Say hello"
}
reward = engine.compute(context)
print(f"Reward: {reward}")

text

## ✨ Key Features

- **🧩 Modular Design**: 37+ reward components across 7 categories
- **🤖 Neural Models**: MLP, Ensemble, and Bayesian reward models
- **🎯 RLHF Support**: Complete preference learning pipeline
- **🧠 Memory Systems**: Episodic and working memory with compression
- **📊 Bayesian Optimization**: Automated hyperparameter tuning
- **📈 Advanced Logging**: Comprehensive metrics and analytics
- **⚡ Production Ready**: Battle-tested, scalable architecture

## 🏗️ Architecture

graph TD
A[Context] --> B[RewardEngine]
B --> C[RewardComposer]
C --> D[Component 1]
C --> E[Component 2]
C --> F[Component N]
D --> G[RewardShaper]
E --> G
F --> G
G --> H[Final Reward]

text

## 📦 Component Categories

| Category | Components | Use Cases |
|----------|------------|-----------|
| **Basic** | Length, Constant, Threshold, Range | Fundamental reward shaping |
| **LLM** | Perplexity, Toxicity, Creativity, Factuality | Language model training |
| **Learning** | Adaptive, Adversarial, Meta-learning | Advanced RL techniques |
| **Robotics** | Collision, Energy, Task completion | Robot control |
| **Advanced** | Multi-objective, Hierarchical, Contrastive | Complex scenarios |

## 📚 Cookbooks & Tutorials

Comprehensive step-by-step guides for real-world applications:

### 🎮 **Reinforcement Learning Integration**
- **[Gym + Stable Baselines3](./examples/cookbooks/gym_stable_baselines3.py)** - Classic RL environments (FrozenLake, CartPole, Atari)
  - Custom reward wrappers for any Gym environment
  - Multi-objective reward optimization
  - Detailed reward component analysis with visualizations

### 🤖 **RLHF & Preference Learning**
- **[Advanced RLHF Training](./examples/cookbooks/advanced_rlhf_training.py)** - Complete preference learning pipeline
  - Preference data collection and management
  - Ensemble models with uncertainty quantification
  - Active learning for efficient data collection
  - Real-world RLHF integration patterns

### 🧠 **Neural Reward Models**
- **[Deep Reward Models](./examples/advanced/deep_reward_model.py)** - Neural network reward learning
  - MLP, Ensemble, and Bayesian models
  - Comprehensive training and evaluation
  - Uncertainty estimation and robustness analysis

### 🔧 **Advanced Techniques**
- **[Hyperparameter Optimization](./examples/cookbooks/hyperparameter_optimization.py)** *(Coming Soon)*
  - Bayesian optimization for reward tuning
  - Multi-objective parameter search
  - Automated configuration generation

- **[Multi-Agent Environments](./examples/cookbooks/multi_agent_rewards.py)** *(Coming Soon)*
  - Competitive and cooperative scenarios
  - Social reward functions
  - Emergent behavior analysis

- **[Custom Environment Creation](./examples/cookbooks/custom_environments.py)** *(Coming Soon)*
  - Building domain-specific environments
  - Advanced reward engineering patterns
  - Production deployment strategies

## 🔧 Configuration

Create a `config.yaml` file:

reward_components:

name: LengthReward
params:
target_length: 100
strength: 0.01

name: DiversityReward
params:
history_size: 10
strength: 0.5

shaping_config:
LengthReward: 1.0
DiversityReward: 0.5

logging:
log_dir: "./logs"
log_frequency: 100

text

## 🎯 Use Cases

### Language Model Training
RLHF with preference learning
from rllama import PreferenceTrainer, MLPRewardModel

model = MLPRewardModel(input_dim=768, hidden_dims=)
trainer = PreferenceTrainer(model)
trainer.train(preference_data)

text

### Robotics Control
Multi-objective robot reward
from rllama import CollisionAvoidanceReward, EnergyEfficiencyReward

components = [
CollisionAvoidanceReward(safety_margin=0.5),
EnergyEfficiencyReward(efficiency_weight=0.3)
]

text

### Game AI
Curriculum learning with adaptive rewards
from rllama import GradualCurriculumReward, ProgressReward

curriculum = GradualCurriculumReward(
initial_difficulty=0.1,
adaptation_rate=0.01
)

text

## 🛠️ Advanced Features

### Bayesian Optimization
from rllama import BayesianRewardOptimizer

optimizer = BayesianRewardOptimizer(
param_space={"weight": (0.1, 2.0)},
eval_function=evaluate_reward
)
results = optimizer.optimize(n_trials=100)

text

### Memory Systems
from rllama import EpisodicMemory, WorkingMemory

episodic = EpisodicMemory(capacity=10000)
working = WorkingMemory(max_size=5)

text

### Ensemble Models
from rllama import EnsembleRewardModel

ensemble = EnsembleRewardModel(
input_dim=512,
num_models=5,
hidden_dims=
)

text

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

git clone https://github.com/ch33nchan/RLlama.git
cd RLlama
pip install -e ".[dev]"

text

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch, Optuna, and NumPy
- Inspired by the RL and LLM communities
- Special thanks to all contributors

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ch33nchan/RLlama/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ch33nchan/RLlama/discussions)
- **Documentation**: [Full Documentation](https://rllama.ai/docs)

---

<div align="center">
  <strong>Built with ❤️ for the RL community</strong>
</div>
