---
id: glossary
title: Glossary of Terms
sidebar_label: Glossary
slug: /glossary
---

# RLlama Glossary

This glossary defines key terms used throughout the RLlama documentation.

## A

### Agent
An entity that observes states and takes actions in an environment to maximize cumulative rewards.

## B

### BaseReward
The foundational class in RLlama that all reward components inherit from.

### Bayesian Optimization
A technique for efficiently optimizing reward component weights using probabilistic models.

## C

### Component
An individual module in RLlama that calculates one aspect of the total reward.

### Context Object
A dictionary containing all information needed to calculate rewards, including state, action, and custom data.

### Curriculum Learning
A training approach where task difficulty increases progressively as the agent improves.

## E

### Episodic Memory
A memory system that stores and retrieves complete episodes or experiences.

## R

### Reward Engineering
The process of designing reward functions to guide reinforcement learning agents.

### Reward Function
A mapping from state and action to a numerical reward value that guides agent learning.

### Reward Hacking
When agents exploit loopholes to maximize rewards without achieving intended goals.

### Reward Shaping
Modifying rewards to guide learning without changing the optimal policy.

### RewardEngine
The central component in RLlama that manages reward components and calculates total rewards.

### RewardOptimizer
A tool for automatically finding optimal reward component weights.

## S

### Sparse Reward
A reward signal that is non-zero only in rare situations, making learning difficult.

## W

### Weight
A multiplier applied to a component's output to adjust its influence on the total reward.

### Working Memory
A memory system that maintains state across multiple time steps within an episode.
