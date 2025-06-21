---
id: requirements
title: System Requirements
sidebar_label: Requirements
slug: /installation/requirements
---

# System Requirements for RLlama

RLlama has the following system requirements:

## Python Version

- Python 3.7 or higher

## Required Dependencies

- numpy>=1.20.0
- pyyaml>=5.1
- matplotlib>=3.3.0 (for visualization)

## Optional Dependencies

For full functionality, you may want to install these optional dependencies:

- torch>=1.9.0 (for neural network reward models)
- optuna>=3.0.0 (for reward optimization)
- gym>=0.21.0 (for Gym integration)
- stable-baselines3>=1.0.0 (for Stable Baselines3 integration)

## Hardware Requirements

RLlama itself has minimal hardware requirements. However, the reinforcement learning algorithms you use with RLlama may have more substantial needs, particularly if using deep reinforcement learning techniques.

For basic usage, any modern computer should be sufficient.
