---
id: mountain-car
title: "MountainCar with Sparse Rewards"
sidebar_label: "MountainCar"
slug: /examples/mountain-car
---

# MountainCar Environment with Sparse Rewards

This example demonstrates how RLlama helps solve the classic sparse reward problem in the MountainCar environment.

<div style={{textAlign: 'center', marginBottom: '20px'}}>
  <img src="/img/examples/mountain_car.gif" alt="MountainCar Environment" width="400" />
</div>

## The Sparse Reward Challenge

The MountainCar environment presents a notable challenge in reinforcement learning:

- The car must reach the top of the right hill
- It doesn't have enough power to drive straight up
- It must build momentum by driving back and forth
- The default reward is -1 per timestep until reaching the goal
- There's no reward gradient to guide the learning process

This sparse reward makes learning extremely difficult because the agent receives no feedback on whether it's making progress until it reaches the goal—which is unlikely to happen through random exploration alone.

## RLlama Solution: Dense, Shaped Rewards

We'll use RLlama to create a rich reward system that guides the learning process by providing feedback on progress toward the goal:

1. **Progress Reward** - Rewards the car for reaching new maximum positions
2. **Velocity Reward** - Rewards the car for building up velocity (potential energy)
3. **Oscillation Penalty** - Penalizes wasteful direction changes

## Complete Example

```python
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from rllama import RewardEngine
from rllama.rewards.base import BaseReward
from rllama.integration import GymWrapper

class ProgressReward(BaseReward):
    """Rewards the car for making progress toward the goal (moving right)"""
    def __init__(self, strength=1.0):
        super().__init__()
        self.max_position = -np.inf
        self.strength = strength
    
    def compute(self, context):
        state = context["current_state"]
        position = state[0]
        
        # Check if we've reached a new maximum position
        if position > self.max_position:
            progress = position - self.max_position
            self.max_position = position
            return progress * self.strength
        return 0.0
    
    def reset(self):
        self.max_position = -np.inf

class VelocityReward(BaseReward):
    """Rewards the car for building up velocity (potential energy)"""
    def __init__(self, strength=0.5):
        super().__init__()
        self.strength = strength
    
    def compute(self, context):
        state = context["current_state"]
        velocity = state[1]
        position = state[0]
        
        # Right-moving velocity is better when going uphill
        if position < 0.4:
            return velocity * self.strength
        # Left-moving velocity is better when we need to build momentum
        elif position < -0.4 and velocity < 0:
            return abs(velocity) * self.strength
        return 0.0

class OscillationPenalty(BaseReward):
    """Penalizes the car for frequent direction changes (wasteful)"""
    def __init__(self, penalty=0.1):
        super().__init__()
        self.last_velocity_sign = None
        self.penalty = penalty
    
    def compute(self, context):
        state = context["current_state"]
        velocity = state[1]
        
        if self.last_velocity_sign is None:
            self.last_velocity_sign = np.sign(velocity)
            return 0.0
        
        current_sign = np.sign(velocity)
        if current_sign != 0 and current_sign != self.last_velocity_sign:
            # Direction changed
            self.last_velocity_sign = current_sign
            return -self.penalty
        
        self.last_velocity_sign = current_sign
        return 0.0
    
    def reset(self):
        self.last_velocity_sign = None

# Create the reward engine
engine = RewardEngine()
progress_reward = ProgressReward(strength=10.0)
engine.add_component(progress_reward)
engine.add_component(VelocityReward(strength=0.5))
engine.add_component(OscillationPenalty(penalty=0.1))

# Create a custom wrapper for MountainCar that resets our stateful reward components
class MountainCarWrapper(GymWrapper):
    def __init__(self, reward_engine, mode="replace"):
        super().__init__(reward_engine, mode=mode)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Reset stateful components
        for component in self.reward_engine.components.values():
            if hasattr(component, 'reset'):
                component.reset()
        
        return obs

# Create the environment
env = gym.make('MountainCar-v0')
wrapped_env = MountainCarWrapper(engine, mode="add").wrap(env)

# Create a DQN agent (good for discrete action spaces)
model = DQN(
    "MlpPolicy",
    wrapped_env,
    learning_rate=0.0005,
    buffer_size=50000,
    exploration_fraction=0.2,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=500,
    learning_starts=1000,
    verbose=1
)

# Train the agent
model.learn(total_timesteps=100000)

# Evaluate and visualize the agent
def visualize_mountaincar_episode(model, env, record_state=True):
    obs = env.reset()
    rewards = []
    contributions = []
    states = []
    positions = []
    velocities = []
    actions = []
    
    done = False
    while not done:
        if record_state:
            states.append(obs)
            positions.append(obs[0])
            velocities.append(obs[1])
            
        action, _ = model.predict(obs)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        contributions.append(engine.get_last_contributions())
    
    return rewards, contributions, states, positions, velocities, actions

# Evaluate with and without RLlama rewards
vanilla_env = gym.make('MountainCar-v0')
rewards_vanilla = []
for _ in range(10):
    obs = vanilla_env.reset()
    episode_reward = 0
    done = False
    steps = 0
    while not done and steps < 1000:  # Cap at 1000 steps to avoid infinite loops
        action, _ = model.predict(obs)
        obs, reward, done, info = vanilla_env.step(action)
        episode_reward += reward
        steps += 1
    rewards_vanilla.append(episode_reward)

rewards_rllama = []
for _ in range(10):
    obs = wrapped_env.reset()
    episode_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = wrapped_env.step(action)
        episode_reward += reward
    rewards_rllama.append(episode_reward)

print(f"Average vanilla reward: {np.mean(rewards_vanilla)}")
print(f"Average RLlama reward: {np.mean(rewards_rllama)}")
print(f"Average vanilla steps: {-np.mean(rewards_vanilla)}")  # Negative reward = steps
print(f"Success rate with vanilla rewards: {sum(r > -200 for r in rewards_vanilla) / len(rewards_vanilla)}")
print(f"Success rate with RLlama rewards: {sum(r > -200 for r in rewards_rllama) / len(rewards_rllama)}")

# Visualize a single episode
rewards, contributions, states, positions, velocities, actions = visualize_mountaincar_episode(model, wrapped_env)

# Create plots
plt.figure(figsize=(15, 10))

# Plot component contributions
plt.subplot(2, 2, 1)
progress_rewards = [c.get("ProgressReward", 0) for c in contributions]
velocity_rewards = [c.get("VelocityReward", 0) for c in contributions]
oscillation_penalties = [c.get("OscillationPenalty", 0) for c in contributions]

plt.plot(progress_rewards, label="Progress")
plt.plot(velocity_rewards, label="Velocity")
plt.plot(oscillation_penalties, label="Oscillation")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Components")
plt.legend()

# Plot position over time
plt.subplot(2, 2, 2)
plt.plot(positions)
plt.axhline(y=0.5, color='r', linestyle='-', label="Goal")
plt.xlabel("Step")
plt.ylabel("Position")
plt.title("Car Position Over Time")
plt.legend()

# Plot velocity over time
plt.subplot(2, 2, 3)
plt.plot(velocities)
plt.xlabel("Step")
plt.ylabel("Velocity")
plt.title("Car Velocity Over Time")

# Plot actions over time
plt.subplot(2, 2, 4)
plt.plot(actions, '-o', markersize=3)
plt.xlabel("Step")
plt.ylabel("Action (0=Left, 1=Nothing, 2=Right)")
plt.yticks([0, 1, 2])
plt.title("Actions Over Time")

plt.tight_layout()
plt.show()

# Create phase space plot (position vs. velocity)
plt.figure(figsize=(10, 6))
plt.scatter(positions, velocities, c=range(len(positions)), cmap='viridis', alpha=0.7)
plt.colorbar(label="Time Step")
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.title("Phase Space Trajectory")
plt.grid(True)
plt.show()

# Record a video
from gym.wrappers import RecordVideo

record_env = gym.make('MountainCar-v0', render_mode="rgb_array")
record_env = RecordVideo(record_env, "videos/mountaincar-rllama")
record_env = MountainCarWrapper(engine, mode="add").wrap(record_env)

obs = record_env.reset()
for _ in range(3):  # Record 3 episodes
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = record_env.step(action)
    obs = record_env.reset()

record_env.close()
```

## Solving the Sparse Reward Problem

The key innovation in this example is using RLlama to transform a sparse reward problem into a dense reward problem:

### Default Reward System
- -1 for each timestep
- 0 upon reaching the goal
- No feedback on progress

### RLlama Enhanced Reward System
- **Progress Reward**: Positive feedback when reaching new positions
- **Velocity Reward**: Guiding the agent to build momentum
- **Oscillation Penalty**: Discouraging wasteful behavior

## Component Breakdown

### ProgressReward

```python
class ProgressReward(BaseReward):
    def __init__(self, strength=1.0):
        super().__init__()
        self.max_position = -np.inf
        self.strength = strength
    
    def compute(self, context):
        state = context["current_state"]
        position = state[0]
        
        # Check if we've reached a new maximum position
        if position > self.max_position:
            progress = position - self.max_position
            self.max_position = position
            return progress * self.strength
        return 0.0
```

This component keeps track of the maximum position reached and rewards the agent for exceeding it. This creates a "breadcrumb trail" of rewards leading toward the goal.

### VelocityReward

```python
class VelocityReward(BaseReward):
    def compute(self, context):
        state = context["current_state"]
        velocity = state[1]
        position = state[0]
        
        # Right-moving velocity is better when going uphill
        if position < 0.4:
            return velocity * self.strength
        # Left-moving velocity is better when we need to build momentum
        elif position < -0.4 and velocity < 0:
            return abs(velocity) * self.strength
        return 0.0
```

This component rewards the agent for building up velocity, which is crucial for gathering enough momentum to reach the top of the hill.

### OscillationPenalty

```python
class OscillationPenalty(BaseReward):
    def compute(self, context):
        state = context["current_state"]
        velocity = state[1]
        
        if self.last_velocity_sign is None:
            self.last_velocity_sign = np.sign(velocity)
            return 0.0
        
        current_sign = np.sign(velocity)
        if current_sign != 0 and current_sign != self.last_velocity_sign:
            # Direction changed
            self.last_velocity_sign = current_sign
            return -self.penalty
```

This component discourages frequent changes in direction that don't contribute to building momentum.

## Stateful Components and Reset

Notice how our components maintain internal state across steps:

```python
def reset(self):
    self.max_position = -np.inf
```

The `reset` method ensures that stateful components are properly initialized at the beginning of each episode.

## Performance Comparison

When comparing agents trained with the default sparse reward versus RLlama's dense rewards:

1. **Learning Speed**: The RLlama agent typically learns to solve the task in far fewer episodes
2. **Success Rate**: A higher percentage of successful episodes
3. **Efficiency**: More consistent and often shorter solution paths

## Visualizations

The phase space plot (position vs. velocity) is particularly revealing:

- It shows how the agent learns to build momentum by swinging back and forth
- Successful strategies create a distinctive pattern of increasing oscillations
- We can observe the agent's strategy of building up potential energy before making the final push

## Key Takeaways

This example demonstrates how RLlama can:

1. **Convert Sparse to Dense Rewards**: Transform difficult sparse reward problems into more learnable dense reward problems
2. **Encode Domain Knowledge**: Incorporate understanding of physics (momentum, energy) into the reward system
3. **Provide Learning Guidance**: Help the agent discover solutions that would be unlikely through random exploration
4. **Track Progress**: Maintain state across steps to measure improvement

## Next Steps

Try experimenting with:

1. Different strengths for each component
2. Additional reward components (e.g., for energy efficiency)
3. Different RL algorithms to compare their performance with shaped rewards
4. Applying similar reward shaping techniques to other sparse reward environments

Check out the [Custom Maze Environment](/docs/examples/maze-environment) example next to see how RLlama works with custom environments.

## Download

You can download the complete notebook for this example:

<a href="/notebooks/mountain_car_example.ipynb" download>Download Jupyter Notebook</a>
