# Quickstart Guide

This guide will get you up and running with RLlama in minutes.

## Basic Setup

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward, DiversityReward

# Create a new reward engine
engine = RewardEngine()

# Add reward components
engine.add_component(LengthReward(target_length=100))
engine.add_component(DiversityReward(history_size=5))

# Compute rewards
context = {
    "response": "This is a sample response",
    "history": ["Previous response 1", "Previous response 2"]
}
reward = engine.compute(context)
print(f"Total reward: {reward}")
```

## Using Weights

```python
# Set custom weights
engine.set_weights({
    "LengthReward": 0.5,
    "DiversityReward": 2.0
})

# Compute reward with new weights
reward = engine.compute(context)
```
