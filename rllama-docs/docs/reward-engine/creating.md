---
id: creating
title: "Creating and Configuring a RewardEngine"
sidebar_label: "Creating RewardEngine"
slug: /reward-engine/creating
---

# Creating and Configuring a RewardEngine

The RewardEngine is the central component of RLlama that orchestrates the calculation of rewards by combining outputs from individual reward components.

## Basic Creation

Creating a RewardEngine is straightforward:

```python
from rllama import RewardEngine

# Create a basic reward engine
engine = RewardEngine()
```

This creates an empty reward engine with default settings. You can then add components to this engine as needed.

## Configuration Options

You can customize the RewardEngine's behavior by passing configuration options:

```python
# Create a reward engine with specific configuration
engine = RewardEngine(
    default_weight=1.0,           # Default weight for components
    normalize_weights=True,       # Automatically normalize weights
    clip_rewards=(-10.0, 10.0),   # Clip final rewards to this range
    use_cache=True,               # Enable reward caching
    cache_size=1000               # Maximum cache entries
)
```

### Available Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `default_weight` | float | 1.0 | Default weight to apply to components that don't have a specific weight set |
| `normalize_weights` | bool | False | If True, automatically normalize weights to sum to 1.0 |
| `clip_rewards` | tuple | None | Range to clip final rewards (min, max), or None for no clipping |
| `use_cache` | bool | False | Enable caching of component results for identical inputs |
| `cache_size` | int | 1000 | Maximum number of entries in the cache |
| `log_level` | str | "INFO" | Logging level ("DEBUG", "INFO", "WARNING", "ERROR") |

## Creating from Configuration Files

For reproducible experiments, you can create a RewardEngine from a YAML configuration file:

```yaml
# reward_config.yaml
engine_config:
  default_weight: 1.0
  normalize_weights: true
  clip_rewards: [-5.0, 5.0]
  use_cache: true
  
reward_components:
  - name: LengthReward
    params:
      target_length: 100
      strength: 0.5
    weight: 1.0
    
  - name: DiversityReward
    params:
      history_size: 5
      strength: 1.0
    weight: 2.0
```

```python
from rllama import RewardEngine
from rllama.utils import load_config

# Load configuration from file
config = load_config('reward_config.yaml')

# Create engine from config
engine = RewardEngine.from_config(config)
```

## Working with Multiple Engines

In more complex scenarios, you might need multiple reward engines for different aspects of your task:

```python
# Create separate engines for different aspects
navigation_engine = RewardEngine()
safety_engine = RewardEngine(clip_rewards=(-float('inf'), 0.0))  # Safety only provides penalties
efficiency_engine = RewardEngine(normalize_weights=True)  # Normalize efficiency-related weights

# Configure each engine separately
navigation_engine.add_component(GoalReward())
safety_engine.add_component(CollisionPenalty())
efficiency_engine.add_component(EnergyEfficiencyReward())
```

These engines can be used independently or combined hierarchically (see [Hierarchical Reward Systems](/docs/advanced/hierarchical)).

## Serialization and Persistence

You can save and load RewardEngine instances to ensure reproducibility:

```python
# Save engine configuration to a file
engine.save_config('my_engine_config.yaml')

# Later, load the configuration
loaded_engine = RewardEngine.from_config('my_engine_config.yaml')
```

## Next Steps

Now that you've created a RewardEngine, you can:

1. [Add reward components](/docs/reward-engine/adding-components) to define the behaviors you want to reward
2. [Set component weights](/docs/reward-engine/setting-weights) to control their relative importance
3. [Compute rewards](/docs/reward-engine/computing-rewards) for your agent's actions
4. [Analyze component contributions](/docs/reward-engine/analyzing-contributions) to understand what drives the rewards
