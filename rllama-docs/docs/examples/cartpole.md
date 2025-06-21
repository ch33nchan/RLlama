---
id: cartpole
title: "CartPole Environment Example"
sidebar_label: "CartPole"
slug: /examples/cartpole
---

# CartPole Environment Example

This example demonstrates how to use RLlama with the classic CartPole balancing task.

<div style={{textAlign: 'center', marginBottom: '20px'}}>
  <img src="/img/examples/cartpole.gif" alt="CartPole Environment" width="400" />
</div>

## Overview

In the CartPole environment, a pole is attached to a cart moving along a frictionless track. The goal is to prevent the pole from falling over by moving the cart left or right. This is a classic control problem and a good starting point for understanding RLlama.

## The Problem with Standard Rewards

The default CartPole environment provides a simple reward function:
- +1 for each time step the pole remains upright
- Episode ends when the pole angle exceeds a threshold or the cart moves too far from the center

This binary reward doesn't give the agent much information about how well it's doing—it only knows whether it failed or not.

## RLlama Solution

With RLlama, we can create a more informative reward system that provides feedback on:
1. How well balanced the pole is (angle)
2. How centered the cart is (position)
3. How stable the system is (minimizing pole and cart movement)

## Complete Example

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from rllama import RewardEngine
from rllama.rewards.base import BaseReward
from rllama.integration import GymWrapper

# Create a custom reward component for balancing
class BalanceReward(BaseReward):
    def __init__(self, angle_threshold=12.0, strength=1.0):
        super().__init__()
        self.angle_threshold = angle_threshold * np.pi / 180  # Convert to radians
        self.strength = strength
    
    def compute(self, context):
        # Get the state (position, velocity, angle, angular_velocity)
        state = context["current_state"]
        angle = state[2]
        
        # Calculate how centered the pole is (angle close to vertical)
        angle_factor = 1.0 - min(abs(angle) / self.angle_threshold, 1.0)
        
        return angle_factor * self.strength

# Create a reward component that encourages keeping the cart centered
class CenteringReward(BaseReward):
    def __init__(self, position_threshold=2.4, strength=0.5):
        super().__init__()
        self.position_threshold = position_threshold
        self.strength = strength
    
    def compute(self, context):
        # Get the state
        state = context["current_state"]
        position = state[0]  # Cart position
        
        # Calculate how centered the cart is
        position_factor = 1.0 - min(abs(position) / self.position_threshold, 1.0)
        
        return position_factor * self.strength

# Create the reward engine
engine = RewardEngine()
engine.add_component(BalanceReward(strength=1.0))
engine.add_component(CenteringReward(strength=0.5))

# Create the environment
env = gym.make('CartPole-v1')

# Wrap the environment with RLlama
wrapped_env = GymWrapper(engine, mode="replace").wrap(env)

# Create a PPO agent
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# Train the agent
model.learn(total_timesteps=25000)

# Evaluate the agent
def evaluate_agent(model, env, num_episodes=10):
    rewards = []
    component_contributions = []
    
    for i in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        episode_contributions = []
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Record component contributions for analysis
            episode_contributions.append(engine.get_last_contributions())
        
        rewards.append(episode_reward)
        component_contributions.append(episode_contributions)
    
    return rewards, component_contributions

rewards, contributions = evaluate_agent(model, wrapped_env)

print(f"Average reward: {np.mean(rewards)}")
print(f"Standard deviation: {np.std(rewards)}")

# Plot rewards per component for the first episode
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
balance_rewards = [c["BalanceReward"] for c in contributions[0]]
centering_rewards = [c["CenteringReward"] for c in contributions[0]]

plt.plot(balance_rewards, label="Balance")
plt.plot(centering_rewards, label="Centering")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Contributions by Component")
plt.legend()

# Plot cumulative rewards
plt.subplot(1, 2, 2)
cumulative_balance = np.cumsum(balance_rewards)
cumulative_centering = np.cumsum(centering_rewards)
plt.plot(cumulative_balance, label="Balance (Cumulative)")
plt.plot(cumulative_centering, label="Centering (Cumulative)")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards by Component")
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Record a video
from gym.wrappers import RecordVideo

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, "videos/cartpole-rllama")
env = GymWrapper(engine, mode="replace").wrap(env)

obs = env.reset()
for _ in range(1000):  # Run for a maximum of 1000 steps
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
env.close()
```

## Step-by-Step Explanation

### 1. Creating Custom Reward Components

First, we define two reward components:

```python
class BalanceReward(BaseReward):
    # Rewards the agent for keeping the pole balanced (small angle)
    
class CenteringReward(BaseReward):
    # Rewards the agent for keeping the cart centered
```

These components provide targeted feedback on specific aspects of the task.

### 2. Setting Up the Reward Engine

```python
engine = RewardEngine()
engine.add_component(BalanceReward(strength=1.0))
engine.add_component(CenteringReward(strength=0.5))
```

We give a higher weight to the BalanceReward (1.0) than the CenteringReward (0.5) since keeping the pole balanced is the primary objective.

### 3. Integrating with Gym

```python
env = gym.make('CartPole-v1')
wrapped_env = GymWrapper(engine, mode="replace").wrap(env)
```

The `mode="replace"` parameter tells RLlama to completely replace the environment's reward function with our custom reward system.

### 4. Training and Evaluation

We train a PPO agent using the wrapped environment and then evaluate its performance, analyzing how each reward component contributes to the total reward.

## Results Analysis

The plots show:

1. **Component Contributions** - How each component (balance and centering) contributes to the reward at each step
2. **Cumulative Rewards** - How the total contribution from each component grows over time

From these, we can observe:
- The BalanceReward provides consistent positive feedback as the pole remains balanced
- The CenteringReward may fluctuate more as the cart moves back and forth
- Over time, the balance component typically contributes more to the total reward

## Key Takeaways

This example demonstrates several core benefits of using RLlama:

1. **Decomposed Objectives** - Breaking down the overall goal into specific behavioral objectives
2. **Transparent Feedback** - Seeing exactly which aspects of behavior are being rewarded
3. **Tunable Importance** - Adjusting the relative importance of different objectives
4. **Enhanced Learning** - Providing richer feedback than the original binary reward

## Next Steps

Try experimenting with:

1. Different component weights to see how they affect the agent's behavior
2. Additional reward components (e.g., for minimizing velocity or jerk)
3. Different RL algorithms (e.g., DQN, A2C) to see how they perform with this reward system

Check out the [LunarLander](/docs/examples/lunar-lander) example next for a more complex environment with more reward components.

## Download

You can download the complete notebook for this example:

<a href="/notebooks/cartpole_example.ipynb" download>Download Jupyter Notebook</a>
