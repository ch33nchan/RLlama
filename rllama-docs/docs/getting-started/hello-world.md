---
id: hello-world
title: '"Hello World" Example'
sidebar_label: "Hello World"
slug: /getting-started/hello-world
---

# RLlama "Hello World" Example

This example demonstrates a complete, minimal working example of RLlama integrated with a simple environment.

## The Cart-Pole Environment

We'll use the classic Cart-Pole environment from OpenAI Gym, where a pole is attached to a cart and the goal is to keep the pole balanced by moving the cart left or right.

## Step 1: Install Dependencies

First, ensure you have all necessary dependencies:

```bash
pip install rllama gym stable-baselines3
```

## Step 2: Create Custom Reward Components

Let's create two custom reward components for the Cart-Pole task:

```python
import numpy as np
from rllama.rewards.base import BaseReward

class BalanceReward(BaseReward):
    """Rewards the agent for keeping the pole balanced (upright)"""
    
    def __init__(self, angle_threshold=12.0, strength=1.0):
        super().__init__()
        self.angle_threshold = angle_threshold * np.pi / 180  # Convert to radians
        self.strength = strength
    
    def compute(self, context):
        # Get current state (position, velocity, angle, angular_velocity)
        state = context["state"]
        angle = state[2]  # Pole angle is at index 2
        
        # Calculate how centered the pole is (angle close to vertical)
        # 1.0 = perfectly upright, 0.0 = at threshold angle
        angle_factor = 1.0 - min(abs(angle) / self.angle_threshold, 1.0)
        
        return angle_factor * self.strength

class CartPositionReward(BaseReward):
    """Rewards the agent for keeping the cart near the center"""
    
    def __init__(self, position_threshold=2.4, strength=0.5):
        super().__init__()
        self.position_threshold = position_threshold
        self.strength = strength
    
    def compute(self, context):
        # Get current state
        state = context["state"]
        position = state[0]  # Cart position is at index 0
        
        # Calculate how centered the cart is (0.0 = at edge, 1.0 = at center)
        position_factor = 1.0 - min(abs(position) / self.position_threshold, 1.0)
        
        return position_factor * self.strength
```

## Step 3: Create the Main Script

Now, let's write the main script that uses RLlama with a reinforcement learning algorithm:

```python
import gym
import numpy as np
from stable_baselines3 import PPO
from rllama import RewardEngine
from rllama.integration import GymWrapper

# Create the reward engine
engine = RewardEngine()
engine.add_component(BalanceReward(strength=1.0))
engine.add_component(CartPositionReward(strength=0.5))

# Create and wrap the environment
env = gym.make('CartPole-v1')
wrapped_env = GymWrapper(engine, mode="add").wrap(env)

# Create the RL agent
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# Train the agent
model.learn(total_timesteps=25000)

# Test the trained agent
obs = wrapped_env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = wrapped_env.step(action)
    total_reward += reward
    
    # Get component-specific information
    contributions = engine.get_last_contributions()
    balance_reward = contributions.get("BalanceReward", 0)
    position_reward = contributions.get("CartPositionReward", 0)
    
    print(f"Step reward: {reward:.2f} (Balance: {balance_reward:.2f}, Position: {position_reward:.2f})")

print(f"Episode complete. Total reward: {total_reward:.2f}")
```

## Step 4: Analyze the Results

When you run this script, you'll see:

1. The agent training with our custom reward components
2. Detailed rewards for each step, broken down by component
3. The total episode reward

Our custom reward components encourage two specific behaviors:
- Keeping the pole balanced (upright)
- Keeping the cart near the center of the track

By using RLlama, we've made these objectives explicit and can see exactly how much each component contributes to the total reward.

## What's Next?

This simple example demonstrates the core functionality of RLlama. From here, you can:

1. Try different reward components or create your own
2. Experiment with component weights to prioritize certain behaviors
3. Apply this approach to more complex environments

For a more advanced example, see the [First Reward System](/docs/getting-started/first-reward-system) tutorial.
