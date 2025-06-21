---
id: overview
title: "RLlama Examples Overview"
sidebar_label: "Examples Overview"
slug: /examples/overview
---

# RLlama Examples Overview

This section contains a collection of example notebooks demonstrating how to use RLlama in various reinforcement learning scenarios. Each example is designed to showcase different features and applications of the library.

## Available Examples

### Basic Examples
- [CartPole Environment](/docs/examples/cartpole) - A simple example using RLlama with the classic CartPole balancing task
- [LunarLander Environment](/docs/examples/lunar-lander) - RLlama applied to the more complex LunarLander environment

### Intermediate Examples
- [MountainCar with Sparse Rewards](/docs/examples/mountain-car) - Using RLlama to solve the sparse reward problem in MountainCar
- [Custom Maze Environment](/docs/examples/maze-environment) - Creating and using a custom environment with RLlama

### Advanced Examples
- [Atari Breakout](/docs/examples/breakout) - Using RLlama with deep RL in Atari Breakout
- [Multi-Objective Navigation](/docs/examples/multi-objective) - Balancing multiple objectives in a navigation task

## Running the Examples

Each example is provided as:

1. A detailed walkthrough in this documentation
2. A downloadable Jupyter notebook
3. Python scripts for running without Jupyter

### Requirements

To run these examples, you'll need:

```
rllama
gym
stable-baselines3
matplotlib
numpy
jupyter (optional, for notebooks)
```

You can install all requirements with:

```bash
pip install "rllama[examples]"
```

### Example Structure

Each example follows a similar structure:

1. **Setup and Environment Creation** - Creating and configuring the environment
2. **Reward Component Design** - Designing custom reward components for the task
3. **Training** - Training an agent using the RLlama reward system
4. **Evaluation** - Analyzing and visualizing the results
5. **Optimization** - Fine-tuning the reward system

## Getting Started

Begin with the [CartPole Environment](/docs/examples/cartpole) example for a gentle introduction to RLlama with a classic reinforcement learning problem.
