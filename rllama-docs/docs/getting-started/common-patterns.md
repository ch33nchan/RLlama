---
id: common-patterns
title: "Common Usage Patterns"
sidebar_label: "Common Patterns"
slug: /getting-started/common-patterns
---

# Common Usage Patterns

This guide covers common patterns and best practices for using RLlama effectively.

## Pattern 1: Stateful Components with Reset

When components need to track state across steps but reset between episodes:

```python
from rllama.rewards.base import BaseReward

class StatefulReward(BaseReward):
    def __init__(self):
        super().__init__()
        self.previous_state = None
    
    def compute(self, context):
        current_state = context["state"]
        
        # First step in episode
        if self.previous_state is None:
            self.previous_state = current_state
            return 0.0
        
        # Calculate based on state change
        reward = self.calculate_improvement(self.previous_state, current_state)
        
        # Update stored state
        self.previous_state = current_state
        
        return reward
    
    def reset(self):
        """Called at the beginning of each episode."""
        self.previous_state = None

# Usage
engine = RewardEngine()
engine.add_component(StatefulReward())

# Reset at episode start
for component in engine.components.values():
    if hasattr(component, 'reset'):
        component.reset()
```

## Pattern 2: Configuration via YAML

For reproducible experiments, define reward systems in configuration files:

```yaml
# reward_config.yaml
reward_components:
  - name: GoalReward
    params:
      strength: 1.0
    weight: 1.0
  
  - name: ObstacleAvoidanceReward
    params:
      safe_distance: 1.0
      penalty: 2.0
    weight: 1.5
  
  - name: EnergyEfficiencyReward
    params:
      efficiency_factor: 0.1
    weight: 0.5
```

```python
from rllama import RewardEngine
from rllama.utils import load_config

# Load configuration
config = load_config('reward_config.yaml')

# Create engine from config
engine = RewardEngine.from_config(config)

# Use as normal
reward = engine.compute(context)
```

## Pattern 3: Gym Environment Integration

Integrating with OpenAI Gym environments:

```python
import gym
from rllama import RewardEngine
from rllama.integration import GymWrapper

# Create environment
env = gym.make('CartPole-v1')

# Create reward engine
engine = RewardEngine()
engine.add_component(BalanceReward())
engine.add_component(CenteringReward())

# Choose integration mode:
# - "replace": Use only RLlama rewards
# - "add": Add RLlama rewards to environment rewards
# - "observe": Use environment rewards but still calculate RLlama rewards
wrapped_env = GymWrapper(engine, mode="replace").wrap(env)

# Use like a standard Gym environment
obs = wrapped_env.reset()
action = policy(obs)
obs, reward, done, info = wrapped_env.step(action)
```

## Pattern 4: Hierarchical Reward Composition

For complex tasks, compose reward hierarchically:

```python
# Create sub-engines for different aspects
navigation_engine = RewardEngine()
navigation_engine.add_component(GoalReward())
navigation_engine.add_component(PathEfficiencyReward())

safety_engine = RewardEngine()
safety_engine.add_component(ObstacleAvoidanceReward())
safety_engine.add_component(StabilityReward())

efficiency_engine = RewardEngine()
efficiency_engine.add_component(EnergyEfficiencyReward())
efficiency_engine.add_component(TimeEfficiencyReward())

# Create wrapper components
class NavigationReward(BaseReward):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
    
    def compute(self, context):
        return self.engine.compute(context)

class SafetyReward(BaseReward):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
    
    def compute(self, context):
        return self.engine.compute(context)

class EfficiencyReward(BaseReward):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
    
    def compute(self, context):
        return self.engine.compute(context)

# Create meta-engine
meta_engine = RewardEngine()
meta_engine.add_component(NavigationReward(navigation_engine))
meta_engine.add_component(SafetyReward(safety_engine))
meta_engine.add_component(EfficiencyReward(efficiency_engine))

# Set high-level weights
meta_engine.set_weights({
    "NavigationReward": 1.0,
    "SafetyReward": 2.0,  # Safety is highest priority
    "EfficiencyReward": 0.5
})
```

## Pattern 5: Dynamic Weight Scheduling

Adjust component weights over time:

```python
from rllama.rewards.scheduler import LinearScheduler

# Create scheduler that linearly changes weight from 2.0 to 0.5 over 1000 steps
scheduler = LinearScheduler(
    initial_weight=2.0,
    final_weight=0.5,
    steps=1000
)

# Add component with scheduler
engine.add_component_with_scheduler(
    ExplorationReward(),
    scheduler,
    name="ExplorationReward"
)

# Update schedulers at each step
for step in range(total_steps):
    # Before computing reward
    engine.update_schedulers(step)
    
    # Then compute reward
    reward = engine.compute(context)
```

## Pattern 6: A/B Testing Reward Configurations

Compare different reward configurations:

```python
# Configuration A
engine_a = RewardEngine()
engine_a.add_component(RewardComponentA(strength=1.0))
engine_a.add_component(RewardComponentB(strength=2.0))

# Configuration B
engine_b = RewardEngine()
engine_b.add_component(RewardComponentA(strength=2.0))
engine_b.add_component(RewardComponentC(strength=1.0))

# Function to evaluate a configuration
def evaluate_config(engine, num_episodes=10):
    total_reward = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, _, done, info = env.step(action)
            
            context = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "done": done,
                "info": info
            }
            
            reward = engine.compute(context)
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

# Compare configurations
score_a = evaluate_config(engine_a)
score_b = evaluate_config(engine_b)

print(f"Configuration A score: {score_a}")
print(f"Configuration B score: {score_b}")
```

## Pattern 7: Logging and Analysis

Track reward components for analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create tracking lists
rewards = []
contributions = []
steps = []

# During training
for step in range(total_steps):
    # ... (training code)
    
    # Compute reward
    reward = engine.compute(context)
    
    # Log data
    rewards.append(reward)
    contributions.append(engine.get_last_contributions())
    steps.append(step)

# Convert to DataFrame for analysis
df = pd.DataFrame(contributions)
df['total_reward'] = rewards
df['step'] = steps

# Plot component contributions over time
plt.figure(figsize=(12, 6))
for column in df.columns:
    if column not in ['total_reward', 'step']:
        plt.plot(df['step'], df[column], label=column)

plt.plot(df['step'], df['total_reward'], 'k--', label='Total Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.title('Reward Component Contributions Over Time')
plt.show()

# Calculate component statistics
stats = df.describe()
print("Component Statistics:")
print(stats)

# Calculate correlations between components
correlations = df.corr()
print("\nComponent Correlations:")
print(correlations)
```

These patterns demonstrate common ways to use RLlama in practice. By adopting these patterns, you can create more effective, maintainable, and transparent reward systems.
