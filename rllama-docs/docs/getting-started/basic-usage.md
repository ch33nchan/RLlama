---
id: basic-usage
title: Basic Usage Tutorial
sidebar_label: Basic Usage
slug: /getting-started/basic-usage
---

# Basic Usage of RLlama

This tutorial introduces the fundamental concepts and basic usage patterns of RLlama.

## Core RLlama Components

Before diving into code, let's understand the core components of RLlama:

1. **RewardEngine**: The central coordinator that manages components and calculates rewards
2. **BaseReward**: The parent class for all reward components
3. **Components**: Individual reward calculators (e.g., LengthReward, DiversityReward)
4. **Context**: A dictionary containing all information needed to calculate rewards

## Your First RLlama Script

Let's write a simple script that demonstrates the core functionality of RLlama:

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward, DiversityReward

# Create a reward engine
engine = RewardEngine()

# Add components that reward specific behaviors
engine.add_component(LengthReward(target_length=50))  # Reward responses close to 50 tokens
engine.add_component(DiversityReward(history_size=3))  # Reward diversity from past responses

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

# Get detailed breakdown of component contributions
contributions = engine.get_last_contributions()
print(f"Component contributions: {contributions}")
```

When you run this script, you'll see the total reward value and the contribution from each component.

## Understanding the Context Object

The context object is a dictionary that provides all the information needed for reward calculation. Different components will look for different keys within this context:

```python
context = {
    # For text generation tasks
    "response": "The generated text",
    "history": ["Previous response 1", "Previous response 2"],
    
    # For reinforcement learning tasks
    "state": current_state,
    "action": selected_action,
    "next_state": resulting_state,
    "done": is_episode_complete,
    
    # Custom information
    "custom_key": custom_value
}
```

You can include any information your components need in the context object.

## Adding Weights to Components

By default, all components have equal weight, but you can adjust their relative importance:

```python
# Set component weights
engine.set_weights({
    "LengthReward": 0.5,    # Less important
    "DiversityReward": 2.0  # More important
})

# Recalculate reward with new weights
reward = engine.compute(context)
print(f"Weighted reward: {reward}")
```

This allows you to balance different aspects of the reward function based on your priorities.

## Next Steps

Now that you understand the basics of RLlama, you can:

1. Explore built-in reward components in the [Components](/docs/components/built-in) section
2. Learn to create custom reward components in the [Custom Components](/docs/components/custom) section
3. See how to integrate RLlama with RL frameworks in the [Integration](/docs/integration/gym) section

For a complete working example, see the [Hello World](/docs/getting-started/hello-world) tutorial.
