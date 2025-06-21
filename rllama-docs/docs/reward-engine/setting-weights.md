---
id: setting-weights
title: "Setting and Adjusting Weights"
sidebar_label: "Setting Weights"
slug: /reward-engine/setting-weights
---

# Setting and Adjusting Component Weights

Weights allow you to control the relative importance of different reward components. This page explains how to set and adjust weights effectively.

## Understanding Weights

Each component in a RewardEngine has a weight that determines its contribution to the total reward:

```
total_reward = (weight_1 * component_1_reward) + (weight_2 * component_2_reward) + ...
```

By default, all components have a weight of 1.0, meaning they contribute equally (relative to their raw output values).

## Setting Individual Weights

To set the weight for a specific component:

```python
# Set weight for a single component
engine.set_component_weight("LengthReward", 0.5)
```

This tells the engine that the LengthReward component should have half the influence it would normally have.

## Setting Multiple Weights at Once

To set weights for multiple components simultaneously:

```python
# Set weights for multiple components
engine.set_weights({
    "LengthReward": 0.5,
    "DiversityReward": 2.0,
    "RelevanceReward": 1.5
})
```

This is more efficient than setting weights individually.

## Getting Current Weights

To check the current weights:

```python
# Get all component weights
weights = engine.get_weights()
print(f"Current weights: {weights}")

# Get specific component weight
length_weight = engine.get_component_weight("LengthReward")
print(f"LengthReward weight: {length_weight}")
```

## Weight Normalization

You can normalize weights to ensure they sum to 1.0:

```python
# Set weights and normalize them
engine.set_weights(
    {
        "LengthReward": 1.0,
        "DiversityReward": 2.0,
        "RelevanceReward": 3.0
    },
    normalize=True
)

# The weights will be adjusted to sum to 1.0:
# LengthReward: 0.167 (1/6)
# DiversityReward: 0.333 (2/6)
# RelevanceReward: 0.500 (3/6)
```

This can be useful when you want to ensure that the total reward scale remains consistent regardless of how many components you add.

## Enabling Weight Normalization by Default

You can configure the RewardEngine to always normalize weights:

```python
# Create engine with automatic weight normalization
engine = RewardEngine(normalize_weights=True)

# All weight updates will be normalized automatically
engine.set_weights({
    "LengthReward": 1.0,
    "DiversityReward": 2.0
})
```

## Copying Weights Between Engines

You can copy weights from one engine to another:

```python
# Create two engines
engine1 = RewardEngine()
engine1.add_component(LengthReward(target_length=100))
engine1.add_component(DiversityReward(history_size=5))
engine1.set_weights({"LengthReward": 0.5, "DiversityReward": 2.0})

engine2 = RewardEngine()
engine2.add_component(LengthReward(target_length=100))
engine2.add_component(DiversityReward(history_size=5))

# Copy weights from engine1 to engine2
weights = engine1.get_weights()
engine2.set_weights(weights)
```

This is useful when experimenting with different component configurations but wanting to maintain the same relative importance.

## Dynamic Weight Adjustment

For more sophisticated weight management, you can adjust weights dynamically during training:

```python
# Define a function to adjust weights based on training progress
def adjust_weights(engine, progress):
    """Adjust weights based on training progress (0.0 to 1.0)"""
    if progress < 0.3:
        # Early training: focus on exploration
        engine.set_weights({
            "ExplorationReward": 2.0,
            "TaskReward": 0.5
        })
    elif progress < 0.7:
        # Mid training: balance exploration and task performance
        engine.set_weights({
            "ExplorationReward": 1.0,
            "TaskReward": 1.0
        })
    else:
        # Late training: focus on task performance
        engine.set_weights({
            "ExplorationReward": 0.5,
            "TaskReward": 2.0
        })

# During training
for episode in range(1000):
    progress = episode / 1000  # Training progress from 0.0 to 1.0
    adjust_weights(engine, progress)
    
    # Train with the current weights
    # ...
```

## Using Weight Schedulers

For smoother weight transitions, RLlama provides weight schedulers:

```python
from rllama.rewards.scheduler import LinearScheduler, ExponentialScheduler

# Create schedulers
linear_scheduler = LinearScheduler(
    initial_weight=2.0,
    final_weight=0.5,
    steps=1000
)

exponential_scheduler = ExponentialScheduler(
    initial_weight=0.1,
    final_weight=1.0,
    steps=1000,
    exponent=2.0
)

# Add components with schedulers
engine.add_component_with_scheduler(
    ExplorationReward(),
    linear_scheduler,
    name="ExplorationReward"
)

engine.add_component_with_scheduler(
    TaskReward(),
    exponential_scheduler,
    name="TaskReward"
)

# During training, update schedulers at each step
for step in range(2000):
    engine.update_schedulers(step)
    
    # Train with the current weights
    # ...
    
    # You can check the current weights
    if step % 200 == 0:
        print(f"Step {step} weights: {engine.get_weights()}")
```

## Automatic Weight Optimization

Instead of manually setting weights, you can use the `RewardOptimizer` to find optimal weights automatically:

```python
from rllama.rewards.optimizer import RewardOptimizer

# Define an evaluation function
def evaluate_weights(weights):
    """Evaluate a set of weights and return a performance score."""
    engine.set_weights(weights)
    
    # Run evaluation episodes
    total_score = 0
    for _ in range(10):
        score = run_evaluation_episode(engine)
        total_score += score
    
    return total_score / 10

# Create optimizer
optimizer = RewardOptimizer(engine)

# Define search space for weights
search_space = {
    "LengthReward": (0.1, 3.0),
    "DiversityReward": (0.5, 5.0),
    "RelevanceReward": (0.5, 5.0)
}

# Run optimization
best_weights = optimizer.optimize(
    evaluate_weights,
    n_trials=100,
    search_space=search_space
)

print(f"Best weights found: {best_weights}")

# Apply the optimized weights
engine.set_weights(best_weights)
```

For more details on weight optimization, see the [Optimization](/docs/optimization/reward-optimizer) section.

## Best Practices for Setting Weights

1. **Start with equal weights** (1.0 for all components) to understand baseline behavior

2. **Adjust incrementally** and observe the effects on agent behavior

3. **Consider component scales** - if a component typically outputs values in a different range than others, adjust its weight or internal scaling accordingly

4. **Use normalization** when the absolute scale of rewards is important

5. **Automate weight tuning** for complex systems with many components

6. **Document your weight choices** to ensure reproducibility and make it easier to understand what each component contributes

By carefully adjusting component weights, you can fine-tune your reward system to effectively guide agent behavior toward your desired objectives.
