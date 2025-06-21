# Getting Started with RLlama

This guide will walk you through the fundamentals of RLlama, from installation to creating your first composable reward system.

## 1. Installation

First, install RLlama using pip. We recommend also installing the `gym` extras to follow along with this guide.

```bash
pip install "rllama[gym]"

2. The Core Idea in 3 Steps
The entire RLlama workflow can be summarized in three simple steps: Define, Compose, and Integrate.

Step 1: Define Your Reward Components
A Reward Component is a simple Python class that focuses on a single aspect of behavior. Let's create two components for the CartPole environment: one to keep the cart centered and one to keep the pole upright.

Python

from rllama.rewards.base import BaseReward

# Component to reward the agent for staying near the center
class CartPositionReward(BaseReward):
    def compute(self, context):
        # The 'context' dict contains all state info
        cart_position = context['state'][0]
        # Reward is higher the closer to the center (0.0)
        return (2.4 - abs(cart_position)) / 2.4

# Component to reward the agent for keeping the pole upright
class PoleAngleReward(BaseReward):
    def compute(self, context):
        pole_angle = context['state'][2]
        # Reward is higher the closer the angle is to 0
        return (0.209 - abs(pole_angle)) / 0.209
Step 2: Compose Components in the Engine
The RewardEngine is the central orchestrator. You add your components to it and can set weights to define their relative importance.

Python

from rllama.engine import RewardEngine

# Create the engine
engine = RewardEngine()

# Add our components
engine.add_component(CartPositionReward())
engine.add_component(PoleAngleReward())

# Set weights to balance the objectives
# Here, we consider them equally important
engine.set_weights({
    "CartPositionReward": 1.0,
    "PoleAngleReward": 1.0,
})
Step 3: Integrate with Your Environment
RLlama provides a GymWrapper to seamlessly integrate your reward logic with any standard Gym environment.

Python

import gym
from rllama.integration import GymWrapper

# Create the standard Gym environment
env = gym.make('CartPole-v1')

# Wrap it with our RLlama engine
wrapped_env = GymWrapper(engine).wrap(env)

# That's it! Now, this environment uses our custom reward logic.
# Let's run a simple loop to see it in action.
obs, info = wrapped_env.reset()
for _ in range(50):
    action = wrapped_env.action_space.sample() # Use a random action
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    
    # The 'reward' is now the composite reward from RLlama
    print(f"RLlama Reward: {reward:.2f}")

    # You can also see the contribution from each component
    print(f"  Contributions: {engine.get_last_contributions()}")
    
    if terminated or truncated:
        obs, info = wrapped_env.reset()

This wrapped_env can now be passed to any RL training library, like Stable Baselines3, for training a proper agent.

You've now successfully built a modular, transparent, and maintainable reward system. From here, you can explore the Cookbook for more advanced examples or the Optimization Guide to learn how to tune your weights automatically.