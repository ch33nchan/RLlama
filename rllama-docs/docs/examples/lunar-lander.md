---
id: lunar-lander
title: "LunarLander Environment Example"
sidebar_label: "LunarLander"
slug: /examples/lunar-lander
---

# LunarLander Environment Example

This example demonstrates how RLlama can be used with the more complex LunarLander environment to create a sophisticated reward system.

<div style={{textAlign: 'center', marginBottom: '20px'}}>
  <img src="/img/examples/lunar_lander.gif" alt="LunarLander Environment" width="400" />
</div>

## Overview

In the LunarLander environment, the agent controls a spacecraft and must land it safely on a landing pad. This is more complex than CartPole because:

1. It has a more complex state space (8 dimensions)
2. It has more actions (4 discrete actions)
3. The dynamics are more challenging
4. The default reward function is already multi-component

## The Default Reward Function

The default LunarLander reward includes:
- Landing on the landing pad: 100-140 points
- Moving away from landing pad: -100 to -70 points
- Leg contact: +10 points each
- Firing the main engine: -0.3 points each frame
- Solved when getting 200+ points

## RLlama Enhancement

We'll use RLlama to create a more structured and transparent reward system with components that focus on:

1. **Stability** - Keeping the lander stable (low angular velocity)
2. **Alignment** - Aligning with the landing pad (horizontal position and angle)
3. **Fuel Efficiency** - Minimizing fuel usage (action penalties)

## Complete Example

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from rllama import RewardEngine
from rllama.rewards.base import BaseReward
from rllama.integration import GymWrapper
from rllama.rewards.optimizer import RewardOptimizer

# Create custom reward components for lunar lander
class StabilityReward(BaseReward):
    """Rewards the lander for being stable (low angular velocity)"""
    def __init__(self, stability_factor=1.0):
        super().__init__()
        self.stability_factor = stability_factor
    
    def compute(self, context):
        state = context["current_state"]
        angular_velocity = state[5]  # Angular velocity is at index 5
        
        # Calculate stability - closer to 0 angular velocity is better
        stability = 1.0 / (1.0 + abs(angular_velocity))
        
        return stability * self.stability_factor

class LandingAlignmentReward(BaseReward):
    """Rewards the lander for being aligned with the landing pad"""
    def __init__(self, alignment_factor=1.0):
        super().__init__()
        self.alignment_factor = alignment_factor
    
    def compute(self, context):
        state = context["current_state"]
        x_position = state[0]  # Horizontal position
        angle = state[4]       # Angle
        
        # Reward being horizontally centered (x close to 0)
        position_alignment = 1.0 / (1.0 + abs(x_position))
        
        # Reward being vertically oriented (angle close to 0)
        angle_alignment = 1.0 / (1.0 + abs(angle))
        
        # Combine alignments
        total_alignment = (position_alignment + angle_alignment) / 2
        
        return total_alignment * self.alignment_factor

class FuelEfficiencyReward(BaseReward):
    """Penalizes fuel usage (actions 2 and 3 use the main and side engines)"""
    def __init__(self, fuel_penalty=0.03):
        super().__init__()
        self.fuel_penalty = fuel_penalty
    
    def compute(self, context):
        action = context["action"]
        
        # Actions 2 and 3 use fuel (main engine and right engine)
        # Action 1 is left engine
        fuel_used = 0
        if action == 2 or action == 3 or action == 1:
            fuel_used = 1
        
        return -fuel_used * self.fuel_penalty

# Create the reward engine
engine = RewardEngine()
engine.add_component(StabilityReward(stability_factor=0.5))
engine.add_component(LandingAlignmentReward(alignment_factor=0.8))
engine.add_component(FuelEfficiencyReward(fuel_penalty=0.03))

# Create the environment
env = gym.make('LunarLander-v2')

# Wrap the environment with RLlama
# Use "add" mode to add our rewards to original rewards
wrapped_env = GymWrapper(engine, mode="add").wrap(env)

# Create the agent
model = A2C("MlpPolicy", wrapped_env, verbose=1)

# Train the agent
model.learn(total_timesteps=200000)

# Optimize reward weights to improve performance
def evaluate_weights(weights):
    # Set new weights
    engine.set_weights(weights)
    
    # Evaluate agent with these weights
    test_env = GymWrapper(engine, mode="add").wrap(gym.make('LunarLander-v2'))
    
    total_rewards = 0
    episodes = 5
    
    for _ in range(episodes):
        obs = test_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward
        
        total_rewards += episode_reward
    
    return total_rewards / episodes

# Create optimizer
optimizer = RewardOptimizer(engine)

# Define search space
search_space = {
    "StabilityReward": (0.1, 2.0),
    "LandingAlignmentReward": (0.3, 3.0),
    "FuelEfficiencyReward": (0.01, 0.2)
}

# Run optimization
best_weights = optimizer.optimize(evaluate_weights, n_trials=30, search_space=search_space)
print(f"Best weights found: {best_weights}")

# Apply optimized weights
engine.set_weights(best_weights)

# Visualize and evaluate
def visualize_episode(model, env):
    obs = env.reset()
    rewards = []
    contributions = []
    states = []
    
    done = False
    while not done:
        states.append(obs)
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        contributions.append(engine.get_last_contributions())
    
    return rewards, contributions, states

env = gym.make('LunarLander-v2', render_mode="rgb_array")
env = GymWrapper(engine, mode="add").wrap(env)

rewards, contributions, states = visualize_episode(model, env)

# Plot rewards and positions
plt.figure(figsize=(15, 10))

# Plot reward components
plt.subplot(2, 2, 1)
stability_rewards = [c["StabilityReward"] for c in contributions]
alignment_rewards = [c["LandingAlignmentReward"] for c in contributions]
fuel_rewards = [c["FuelEfficiencyReward"] for c in contributions]

plt.plot(stability_rewards, label="Stability")
plt.plot(alignment_rewards, label="Alignment")
plt.plot(fuel_rewards, label="Fuel Efficiency")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Components")
plt.legend()

# Plot x, y positions
plt.subplot(2, 2, 2)
x_positions = [state[0] for state in states]
y_positions = [state[1] for state in states]
plt.plot(x_positions, y_positions)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Landing Trajectory")
plt.grid(True)

# Plot angles
plt.subplot(2, 2, 3)
angles = [state[4] for state in states]
plt.plot(angles)
plt.xlabel("Step")
plt.ylabel("Angle (radians)")
plt.title("Lander Angle Over Time")
plt.grid(True)

# Plot cumulative reward
plt.subplot(2, 2, 4)
plt.plot(np.cumsum(rewards))
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Total Reward")
plt.grid(True)

plt.tight_layout()
plt.show()

# Record a video
from gym.wrappers import RecordVideo

record_env = gym.make('LunarLander-v2', render_mode="rgb_array")
record_env = RecordVideo(record_env, "videos/lunarlander-rllama")
record_env = GymWrapper(engine, mode="add").wrap(record_env)

obs = record_env.reset()
for _ in range(3):  # Record 3 episodes
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = record_env.step(action)
    obs = record_env.reset()

record_env.close()
```

## Step-by-Step Explanation

### 1. Creating Custom Reward Components

We create three reward components that focus on different aspects of the landing task:

```python
class StabilityReward(BaseReward):
    # Rewards low angular velocity
    
class LandingAlignmentReward(BaseReward):
    # Rewards correct position and orientation
    
class FuelEfficiencyReward(BaseReward):
    # Penalizes fuel usage
```

### 2. Setting Up the Reward Engine

```python
engine = RewardEngine()
engine.add_component(StabilityReward(stability_factor=0.5))
engine.add_component(LandingAlignmentReward(alignment_factor=0.8))
engine.add_component(FuelEfficiencyReward(fuel_penalty=0.03))
```

### 3. Integration Mode: Adding to Original Rewards

```python
wrapped_env = GymWrapper(engine, mode="add").wrap(env)
```

We use `mode="add"` to add our custom rewards to the environment's original rewards, rather than replacing them. This allows us to enhance the reward signal while still keeping the essential task-completion rewards.

### 4. Weight Optimization

A key feature demonstrated in this example is automatic weight optimization:

```python
optimizer = RewardOptimizer(engine)
best_weights = optimizer.optimize(evaluate_weights, n_trials=30, search_space=search_space)
```

This uses Bayesian optimization to find the best component weights for maximizing performance.

## Results Analysis

The visualizations provide insights into the landing process:

1. **Reward Components** - Shows how each aspect contributes to the total reward
2. **Landing Trajectory** - Visualizes the path taken by the lander
3. **Angle Over Time** - Shows how the lander's orientation changes
4. **Cumulative Reward** - Shows the total reward accumulation

## Comparing With and Without RLlama

We can run a comparative analysis between the default reward and our enhanced RLlama reward system:

```python
# Code for comparative evaluation
standard_env = gym.make('LunarLander-v2')
rllama_env = GymWrapper(engine, mode="add").wrap(gym.make('LunarLander-v2'))

# Train agents on each environment
standard_model = A2C("MlpPolicy", standard_env, verbose=0)
standard_model.learn(total_timesteps=100000)

rllama_model = A2C("MlpPolicy", rllama_env, verbose=0)
rllama_model.learn(total_timesteps=100000)

# Evaluate both models
def evaluate(model, env, episodes=100):
    scores = []
    for _ in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        scores.append(total_reward)
    return np.mean(scores), np.std(scores)

standard_score = evaluate(standard_model, standard_env)
rllama_score = evaluate(rllama_model, standard_env)  # Evaluate on standard env for fair comparison

print(f"Standard reward: mean={standard_score[0]:.2f}, std={standard_score[1]:.2f}")
print(f"RLlama reward: mean={rllama_score[0]:.2f}, std={rllama_score[1]:.2f}")
```

Typically, the RLlama-enhanced agent learns faster and achieves better stability during landing.

## Key Takeaways

This example demonstrates:

1. **Enhanced Reward Signals** - Providing more detailed feedback beyond the default rewards
2. **Multi-objective Balancing** - Managing stability, alignment, and efficiency simultaneously
3. **Automatic Optimization** - Finding optimal component weights through Bayesian optimization
4. **Visualization** - Understanding the contribution of each reward component

## Next Steps

Try experimenting with:

1. Different reward components (e.g., for landing velocity or leg contact)
2. Different integration modes (`replace` instead of `add`)
3. More sophisticated reward optimization strategies
4. Applying similar techniques to other continuous control tasks

Check out the [MountainCar](/docs/examples/mountain-car) example next to see how RLlama can help with sparse reward environments.

## Download

You can download the complete notebook for this example:

<a href="/notebooks/lunar_lander_example.ipynb" download>Download Jupyter Notebook</a>
