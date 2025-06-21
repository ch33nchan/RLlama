---
id: verification
title: Verifying Installation
sidebar_label: Verification
slug: /installation/verification
---

# Verifying Your RLlama Installation

After installing RLlama, it's important to verify that everything is working correctly.

## Basic Verification

Run the following Python code to verify that RLlama is installed and can be imported:

```python
import rllama
print(f"RLlama version: {rllama.__version__}")
```

## Testing a Simple Reward Engine

Create a basic reward engine to ensure the core functionality works:

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward  # A built-in component

# Create a reward engine
engine = RewardEngine()

# Add a simple component
engine.add_component(LengthReward(target_length=10))

# Test with a simple context
context = {"response": "Hello world"}
reward = engine.compute(context)

print(f"Reward: {reward}")
print(f"Component contributions: {engine.get_last_contributions()}")
```

If this runs without errors and prints a reward value and component contribution, your installation is working correctly.

## Testing Optional Dependencies

If you've installed optional dependencies, you can verify they're working correctly:

### Testing Gym Integration

```python
try:
    import gym
    from rllama.integration import GymWrapper
    
    # Create a simple environment
    env = gym.make('CartPole-v1')
    
    # Create a reward engine
    engine = RewardEngine()
    
    # Wrap the environment
    wrapped_env = GymWrapper(engine).wrap(env)
    
    # Try a reset and step
    obs = wrapped_env.reset()
    action = wrapped_env.action_space.sample()
    obs, reward, done, info = wrapped_env.step(action)
    
    print("Gym integration is working correctly!")
except ImportError:
    print("Gym integration not installed. Use 'pip install rllama[gym]' to install.")
```

### Testing Visualization Tools

```python
try:
    from rllama.visualization import RewardVisualizer
    import matplotlib.pyplot as plt
    
    # Create a simple visualization
    engine = RewardEngine()
    visualizer = RewardVisualizer(engine)
    
    # This should create a figure object
    fig = visualizer.create_empty_plot()
    
    print("Visualization tools are working correctly!")
except ImportError:
    print("Visualization tools not installed. Use 'pip install rllama[vis]' to install.")
```

## Troubleshooting Common Issues

If you encounter issues:

1. **ImportError**: Make sure you've installed RLlama and all required dependencies
2. **Version conflicts**: Check that your numpy and other packages meet the minimum version requirements
3. **GPU issues**: If using GPU acceleration, verify that PyTorch can access your GPU

For more detailed troubleshooting, see the [Troubleshooting](/docs/troubleshooting) section.
