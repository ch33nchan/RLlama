# API Reference Overview

This page provides an overview of the main classes and functions in RLlama.

## RewardEngine

The central component that manages all reward components.

```python
engine = RewardEngine()
engine.add_component(component)
engine.compute(context)
```

## BaseReward

The base class for all reward components.

```python
class MyReward(BaseReward):
    def compute(self, context):
        return reward_value
```
