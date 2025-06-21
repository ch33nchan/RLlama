---
id: adding-components
title: "Adding Components"
sidebar_label: "Adding Components"
slug: /reward-engine/adding-components
---

# Adding Components to a RewardEngine

Once you've created a RewardEngine, the next step is to add reward components that define what behaviors you want to encourage or discourage.

## Basic Component Addition

You can add components to a RewardEngine using the `add_component` method:

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward, DiversityReward

# Create a reward engine
engine = RewardEngine()

# Add components
engine.add_component(LengthReward(target_length=100))
engine.add_component(DiversityReward(history_size=5))
```

## Component Naming

By default, components are named after their class, but you can specify a custom name:

```python
# Add a component with a custom name
engine.add_component(LengthReward(target_length=50), name="ShortResponseReward")
engine.add_component(LengthReward(target_length=200), name="LongResponseReward")
```

This is useful when adding multiple components of the same class but with different parameters.

## Adding Built-in Components

RLlama includes many built-in components:

```python
from rllama.rewards.components import (
    LengthReward,          # Rewards based on text/response length
    DiversityReward,       # Rewards diversity compared to history
    RelevanceReward,       # Rewards relevance to a prompt or query
    CuriosityReward,       # Rewards exploring new states
    CompletionReward,      # Rewards task completion
    ConstraintReward       # Rewards adhering to constraints
)

# Add built-in components with different configurations
engine.add_component(LengthReward(target_length=100, mode="gaussian"))
engine.add_component(DiversityReward(history_size=5, similarity_threshold=0.7))
engine.add_component(RelevanceReward(embedding_model="text-embedding-3-small"))
```

See the [Built-in Components](/docs/components/built-in) section for a complete list of available components and their parameters.

## Adding Custom Components

You can also add your own custom reward components:

```python
from rllama.rewards.base import BaseReward

class MyCustomReward(BaseReward):
    def __init__(self, parameter1=1.0, parameter2="default"):
        super().__init__()
        self.parameter1 = parameter1
        self.parameter2 = parameter2
    
    def compute(self, context):
        # Custom reward calculation logic
        # ...
        return calculated_reward

# Add custom component
engine.add_component(MyCustomReward(parameter1=2.0, parameter2="custom"))
```

For more details on creating custom components, see the [Custom Components](/docs/components/custom) section.

## Adding Components with Weights

You can specify a weight when adding a component:

```python
# Add component with a specific weight
engine.add_component(LengthReward(target_length=100), weight=0.5)
engine.add_component(DiversityReward(history_size=5), weight=2.0)
```

This is equivalent to adding the component and then setting its weight separately.

## Adding Multiple Components at Once

To add multiple components efficiently:

```python
# Add multiple components at once
engine.add_components([
    LengthReward(target_length=100),
    DiversityReward(history_size=5),
    RelevanceReward()
])
```

## Dynamic Component Addition

You can also add components dynamically during runtime:

```python
def configure_engine_based_on_task(task_type):
    engine = RewardEngine()
    
    # Add base components for all tasks
    engine.add_component(ConstraintReward())
    
    # Add task-specific components
    if task_type == "text_generation":
        engine.add_component(LengthReward(target_length=100))
        engine.add_component(DiversityReward(history_size=5))
    elif task_type == "navigation":
        engine.add_component(GoalReward())
        engine.add_component(ObstacleAvoidanceReward())
    elif task_type == "custom":
        engine.add_component(MyCustomReward())
    
    return engine

# Create task-specific engine
text_engine = configure_engine_based_on_task("text_generation")
nav_engine = configure_engine_based_on_task("navigation")
```

## Adding Components with Schedulers

For dynamic weight adjustment over time, you can add components with schedulers:

```python
from rllama.rewards.scheduler import LinearScheduler

# Create a scheduler that reduces the weight from 2.0 to 0.5 over 1000 steps
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
```

This is useful when you want to gradually change the importance of certain behaviors during training.

## Checking Added Components

You can check what components have been added to your engine:

```python
# Get all components
components = engine.components

# Print component names
print(f"Components: {list(components.keys())}")

# Check if a specific component exists
if "LengthReward" in engine.components:
    print("LengthReward component is present")
```

## Removing Components

You can also remove components if needed:

```python
# Remove a component by name
engine.remove_component("LengthReward")

# Check if removal was successful
if "LengthReward" not in engine.components:
    print("LengthReward component successfully removed")
```

## Next Steps

Now that you've added components to your RewardEngine, you can:

1. [Set component weights](/docs/reward-engine/setting-weights) to control their relative importance
2. [Compute rewards](/docs/reward-engine/computing-rewards) for your agent's actions
3. [Analyze component contributions](/docs/reward-engine/analyzing-contributions) to understand what drives the rewards
