#!/usr/bin/env python3
"""
Example demonstrating the advanced reward components in RLlama.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama import RewardEngine
from rllama.rewards.components import DiversityReward, CuriosityReward, ProgressReward

def main():
    """Run advanced components example"""
    print("RLlama Advanced Components Example")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Test diversity reward
    print("\n1. Diversity Reward")
    print("-" * 40)
    
    diversity_reward = DiversityReward(history_size=5, key="action", strength=1.0)
    
    # Create a sequence of actions with some repetition
    actions = [0, 1, 2, 0, 3, 1, 0, 2, 3, 1]
    rewards = []
    
    for i, action in enumerate(actions):
        context = {"action": action}
        reward = diversity_reward.calculate(context)
        rewards.append(reward)
        print(f"Action {action} (step {i+1}) reward: {reward:.4f}")
    
    # Plot diversity rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards)+1), rewards, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Diversity Reward')
    plt.title('Diversity Reward for Action Sequence')
    plt.grid(True, alpha=0.3)
    plt.savefig("./output/diversity_reward.png")
    
    # Test curiosity reward
    print("\n2. Curiosity Reward")
    print("-" * 40)
    
    curiosity_reward = CuriosityReward(scaling=1.0, decay=0.9)
    
    # Simulate prediction errors that gradually decrease (learning)
    errors = [1.0, 0.8, 0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.02]
    rewards = []
    
    for i, error in enumerate(errors):
        context = {"prediction_error": error}
        reward = curiosity_reward.calculate(context)
        rewards.append(reward)
        print(f"Prediction error {error:.2f} (step {i+1}) reward: {reward:.4f}")
    
    # Plot curiosity rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards)+1), rewards, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Curiosity Reward')
    plt.title('Curiosity Reward with Decaying Scaling')
    plt.grid(True, alpha=0.3)
    plt.savefig("./output/curiosity_reward.png")
    
    # Test progress reward
    print("\n3. Progress Reward")
    print("-" * 40)
    
    progress_reward = ProgressReward(goal_key="goal", state_key="state", scaling=1.0)
    
    # Set a fixed goal
    goal = np.array([10.0, 10.0])
    
    # Create a trajectory that approaches the goal
    states = [
        np.array([0.0, 0.0]),
        np.array([2.0, 1.0]),
        np.array([4.0, 3.0]),
        np.array([5.0, 5.0]),
        np.array([7.0, 6.0]),
        np.array([8.0, 8.0]),
        np.array([9.0, 9.0]),
        np.array([9.5, 9.5]),
        np.array([10.0, 10.0]),
    ]
    
    rewards = []
    distances = []
    
    for i, state in enumerate(states):
        context = {"goal": goal, "state": state}
        reward = progress_reward.calculate(context)
        rewards.append(reward)
        distances.append(np.linalg.norm(state - goal))
        print(f"State {state} (step {i+1}) - distance: {distances[-1]:.4f}, reward: {reward:.4f}")
    
    # Plot progress rewards and distances
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(range(1, len(states)+1), distances, marker='o')
    ax1.set_ylabel('Distance to Goal')
    ax1.set_title('Distance and Progress Reward')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, len(rewards)+1), rewards, marker='o', color='orange')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Progress Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./output/progress_reward.png")
    
    print("\n✅ Advanced components example completed successfully!")

if __name__ == "__main__":
    main()
