---
id: features
title: Key Features and Benefits
sidebar_label: Key Features
slug: /introduction/features
---

# Key Features and Benefits

RLlama offers a comprehensive suite of features designed to make reward engineering more systematic and effective.

## Modular Reward Components

With RLlama, you can design complex reward functions by combining simpler, focused components:

- **Single-Responsibility Components**: Each component focuses on one aspect of behavior
- **Reusable Building Blocks**: Create a library of components you can use across projects
- **Composable Design**: Mix and match components to create sophisticated reward systems

```python
# Example: Combining navigation components
engine.add_component(GoalReward(target_position=[10, 10]))
engine.add_component(ObstacleAvoidanceReward())
engine.add_component(EnergyEfficiencyReward())
```

## Transparent Reward Calculations

RLlama provides complete visibility into how rewards are calculated:

- **Component Contributions**: See exactly how much each component contributes
- **Step-by-Step Breakdown**: Trace reward calculation through each step
- **Debugging Support**: Identify which components are causing issues

```python
reward = engine.compute(context)
contributions = engine.get_last_contributions()
print(f"Total reward: {reward}")
print(f"Component contributions: {contributions}")
# {'GoalReward': 0.8, 'ObstacleAvoidanceReward': -0.2, 'EnergyEfficiencyReward': -0.1}
```

## Automated Optimization

Instead of manually tuning reward weights, RLlama integrates with Optuna for systematic optimization:

- **Bayesian Optimization**: Find optimal weights efficiently 
- **Hyperparameter Search**: Explore component parameters systematically
- **Multi-objective Optimization**: Balance competing objectives

```python
optimizer = RewardOptimizer(engine)
best_weights = optimizer.optimize(evaluate_weights, n_trials=100)
engine.set_weights(best_weights)
```

## Memory Systems

RLlama includes memory components to help agents learn from history:

- **Episodic Memory**: Store and retrieve past experiences
- **Working Memory**: Maintain state across multiple time steps
- **Memory Visualization**: Analyze what's being stored and retrieved

## Seamless Integration

RLlama works with popular reinforcement learning frameworks:

- **OpenAI Gym**: Wrap standard environments to use RLlama rewards
- **Stable Baselines3**: Train agents using popular algorithms
- **Custom Environments**: Integrate with your own environments

```python
# Wrap a Gym environment
env = gym.make('CartPole-v1')
wrapped_env = GymWrapper(engine).wrap(env)

# Use with Stable Baselines3
model = PPO("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=50000)
```

## Visualization Tools

RLlama provides tools for visualizing and understanding reward patterns:

- **Reward History**: Track how rewards change over time
- **Component Analysis**: See the contribution of each component
- **Interactive Dashboards**: Explore patterns in reward distribution

## Designed for Research and Production

RLlama is built to support both research exploration and production deployment:

- **Configurable via YAML**: Define reward systems in portable config files
- **Experiment Tracking**: Log and compare different reward configurations
- **Extensible Architecture**: Add new components and integrations as needed

By providing these tools and abstractions, RLlama transforms reward engineering from a black art into a systematic engineering discipline.
