---
id: computing-rewards
title: "Computing Rewards"
sidebar_label: "Computing Rewards"
slug: /reward-engine/computing-rewards
---

# Computing Rewards with RewardEngine

This guide covers how to use a RewardEngine to calculate rewards for your agent's actions or outputs.

## Basic Reward Computation

Once you've set up your RewardEngine with components, computing rewards is straightforward:

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward, DiversityReward

# Create and configure the reward engine
engine = RewardEngine()
engine.add_component(LengthReward(target_length=50))
engine.add_component(DiversityReward(history_size=3))

# Create a context object with information needed for reward calculation
context = {
    "response": "This is a sample response that we want to evaluate.",
    "history": [
        "Previous response 1",
        "Previous response 2",
        "Previous response 3"
    ]
}

# Calculate reward
reward = engine.compute(context)
print(f"Total reward: {reward}")
```

## The Context Object

The context object is a dictionary that provides all the information needed to calculate rewards. Different components will look for different keys in this context:

### For Text Generation Tasks:

```python
context = {
    "response": "The generated text",
    "prompt": "The input prompt",
    "history": ["Previous response 1", "Previous response 2"],
    "tokens": [token1, token2, token3],  # Tokenized response
    "token_logprobs": [logp1, logp2, logp3]  # Token log probabilities
}
```

### For Reinforcement Learning Tasks:

```python
context = {
    "state": current_state,
    "action": selected_action,
    "reward": environment_reward,  # Original reward from environment
    "next_state": resulting_state,
    "done": is_episode_complete,
    "info": additional_info
}
```

### For Navigation Tasks:

```python
context = {
    "state": {
        "position": [x, y],
        "velocity": [vx, vy],
        "obstacles": [[ox1, oy1], [ox2, oy2], ...],
        "goal_position": [gx, gy]
    },
    "action": [action_x, action_y],
    "next_state": {
        # Same structure as state
    }
}
```

You can include any information your components need in the context object. It's flexible by design.

## Reward Computation Process

When you call `engine.compute(context)`, the following steps occur:

1. The context is passed to each component's `compute` method
2. Each component calculates its individual reward
3. Each component's reward is multiplied by its weight
4. The weighted component rewards are summed to produce the final reward

```
total_reward = (weight_1 * component_1_reward) + (weight_2 * component_2_reward) + ...
```

## Getting Component Contributions

You can see how much each component contributed to the total reward:

```python
# Compute reward
reward = engine.compute(context)

# Get breakdown of component contributions
contributions = engine.get_last_contributions()
print(f"Total reward: {reward}")
print(f"Component contributions: {contributions}")

# Contributions might look like:
# {'LengthReward': 0.85, 'DiversityReward': 1.5}
```

This is invaluable for debugging and understanding what aspects of behavior are being rewarded or penalized.

## Reward Caching

For performance optimization, RLlama can cache reward calculations:

```python
# Create engine with caching enabled
engine = RewardEngine(use_cache=True, cache_size=1000)

# Compute reward (first time: calculated)
reward1 = engine.compute(context)

# Compute reward again with the same context (retrieved from cache)
reward2 = engine.compute(context)
```

This can significantly improve performance when the same or similar contexts are evaluated multiple times.

## Reward Clipping

You can clip rewards to keep them within a desired range:

```python
# Create engine with reward clipping
engine = RewardEngine(clip_rewards=(-10.0, 10.0))

# All computed rewards will be clipped to the range [-10.0, 10.0]
```

This helps prevent extremely large positive or negative rewards that might destabilize learning.

## Computing Rewards in Batches

For efficiency when processing multiple samples:

```python
# List of contexts
contexts = [context1, context2, context3, context4]

# Compute rewards for all contexts
rewards = [engine.compute(ctx) for ctx in contexts]

# Or for more details, get component contributions too
results = []
for ctx in contexts:
    reward = engine.compute(ctx)
    contributions = engine.get_last_contributions()
    results.append((reward, contributions))
```

## Asynchronous Reward Computation

For components that might involve API calls or other asynchronous operations:

```python
import asyncio

# Assuming components support async computation
async def compute_rewards_async(engine, contexts):
    rewards = []
    for ctx in contexts:
        reward = await engine.compute_async(ctx)
        rewards.append(reward)
    return rewards

# Use in an async function
rewards = await compute_rewards_async(engine, contexts)
```

Note: Asynchronous computation requires components that support the `compute_async` method.

## Error Handling

RLlama handles errors in individual components to ensure the reward computation doesn't completely fail:

```python
from rllama.rewards.base import BaseReward

class PotentiallyFailingReward(BaseReward):
    def compute(self, context):
        # This might raise an exception
        if "key" not in context:
            raise KeyError("Required key missing from context")
        return 1.0

# Add to engine
engine.add_component(PotentiallyFailingReward())

# Create a context that will cause the component to fail
bad_context = {"not_the_right_key": "value"}

# This will still compute a reward, excluding the failing component
# (with a warning logged)
reward = engine.compute(bad_context)
```

By default, components that raise exceptions during computation are skipped, and their contributions are set to 0.

## Custom Reward Aggregation

If you need more control over how component rewards are combined:

```python
# Get individual component rewards without aggregating
component_rewards = engine.compute_components(context)
print(f"Component rewards: {component_rewards}")

# Apply custom aggregation
# For example, take the minimum reward instead of the sum
min_reward = min(component_rewards.values())
print(f"Minimum reward: {min_reward}")

# Or apply a different weighting scheme
custom_weights = {"LengthReward": 0.3, "DiversityReward": 0.7}
weighted_rewards = {name: reward * custom_weights.get(name, 1.0) 
                   for name, reward in component_rewards.items()}
custom_total = sum(weighted_rewards.values())
print(f"Custom weighted total: {custom_total}")
```

## Integration with RL Frameworks

When using RLlama with standard RL frameworks, the reward computation is typically incorporated into the environment loop:

```python
# For a basic RL loop
state = env.reset()
done = False

while not done:
    # Select action based on current state
    action = agent.select_action(state)
    
    # Take action in environment
    next_state, env_reward, done, info = env.step(action)
    
    # Create context for reward calculation
    context = {
        "state": state,
        "action": action,
        "next_state": next_state,
        "done": done,
        "info": info,
        "env_reward": env_reward  # Include original environment reward if needed
    }
    
    # Calculate reward using RLlama
    reward = engine.compute(context)
    
    # Update agent with the calculated reward
    agent.update(state, action, reward, next_state, done)
    
    # Move to next state
    state = next_state
```

For more details on integration with specific frameworks, see the [Framework Integration](/docs/integration/gym) section.

## Next Steps

Now that you understand how to compute rewards, you can:

1. [Analyze component contributions](/docs/reward-engine/analyzing-contributions) to understand what drives the rewards
2. [Optimize component weights](/docs/optimization/reward-optimizer) to improve agent performance
3. Learn about more advanced usage in the [Advanced Usage](/docs/advanced/hierarchical) section
