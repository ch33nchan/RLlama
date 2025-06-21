---
id: configuration
title: Configuration Options
sidebar_label: Configuration
slug: /installation/configuration
---

# Configuring RLlama

After installation, you may want to configure RLlama to suit your specific needs. This page covers the various configuration options available.

## Environment Variables

RLlama supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RLLAMA_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `RLLAMA_CONFIG_PATH` | Custom path to configuration files | None |
| `RLLAMA_DISABLE_WARNINGS` | Disable warning messages when set to "1" | 0 |
| `RLLAMA_MAX_MEMORY` | Maximum memory use for caching (MB) | 1000 |

Example usage:

```bash
# Set logging to debug level
export RLLAMA_LOG_LEVEL=DEBUG

# Run your script
python my_rl_script.py
```

## Configuration File

RLlama can load configuration from a YAML file:

```yaml
# rllama_config.yaml
logging:
  level: INFO
  file: logs/rllama.log
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

memory:
  max_size: 1000  # MB
  ttl: 3600  # seconds

optimization:
  default_trials: 100
  parallelism: 4
```

To load this configuration:

```python
from rllama.utils import load_config

# Load configuration from file
config = load_config('path/to/rllama_config.yaml')

# Use config to initialize components
from rllama import RewardEngine
engine = RewardEngine(config=config)
```

## Programmatic Configuration

You can also configure RLlama programmatically:

```python
import rllama

# Configure logging
rllama.set_log_level('DEBUG')

# Configure memory limits
rllama.set_memory_limit(2000)  # 2000 MB

# Configure default optimization settings
from rllama.optimization import set_default_optimization_params
set_default_optimization_params(n_trials=200, timeout=3600)
```

## Component-Specific Configuration

Most RLlama components accept configuration parameters during initialization:

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward, DiversityReward

# Configure individual components
engine = RewardEngine()
engine.add_component(
    LengthReward(
        target_length=100,
        mode='gaussian',
        sigma=20,
        strength=0.5
    )
)
engine.add_component(
    DiversityReward(
        history_size=5,
        similarity_threshold=0.7,
        strength=1.0,
        use_embeddings=True
    )
)
```

## Persistence

You can save and load RLlama configurations:

```python
# Save configuration
engine.save_config('my_reward_config.yaml')

# Load configuration
from rllama import RewardEngine
engine = RewardEngine.from_config('my_reward_config.yaml')
```

This allows you to easily share and reproduce reward configurations across experiments.
