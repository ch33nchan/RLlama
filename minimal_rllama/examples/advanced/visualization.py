#!/usr/bin/env python3
"""
Example demonstrating the visualization tools in RLlama.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama.dashboard.visualizer import RewardVisualizer

def generate_sample_data(num_episodes=5, steps_per_episode=100):
    """Generate sample reward data for visualization."""
    data = []
    
    for episode in range(num_episodes):
        # Base reward improves with each episode
        base_reward = 0.1 * episode
        
        for step in range(steps_per_episode):
            # Generate some noise
            noise = np.random.normal(0, 0.1)
            
            # Total reward increases with steps
            total_reward = base_reward + 0.001 * step + noise
            
            # Component rewards
            length_reward = -0.001 * (step % 50) ** 2
            constant_reward = 0.1
            diversity_reward = 0.05 * np.sin(step / 10)
            
            entry = {
                "step": episode * steps_per_episode + step,
                "episode": episode,
                "total_reward": total_reward,
                "component_rewards": {
                    "LengthReward": length_reward,
                    "ConstantReward": constant_reward,
                    "DiversityReward": diversity_reward
                },
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            data.append(entry)
    
    return data

def main():
    """Run visualization example"""
    print("RLlama Visualization Example")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("./output/logs", exist_ok=True)
    os.makedirs("./output/plots", exist_ok=True)
    
    print("\n1. Generating sample reward data")
    print("-" * 40)
    
    # Generate sample data for two different runs
    run1_data = generate_sample_data(num_episodes=5, steps_per_episode=100)
    run2_data = generate_sample_data(num_episodes=5, steps_per_episode=100)
    
    # Save data to log files
    with open("./output/logs/run1.json", "w") as f:
        json.dump(run1_data, f, indent=2)
    
    with open("./output/logs/run2.json", "w") as f:
        json.dump(run2_data, f, indent=2)
    
    print(f"Generated and saved data for 2 runs")
    
    print("\n2. Loading and visualizing data")
    print("-" * 40)
    
    # Create visualizer
    visualizer = RewardVisualizer("./output/logs")
    
    # Create and save plots
    print("Creating total rewards plot")
    fig = visualizer.plot_total_rewards(smoothing=10)
    fig.savefig("./output/plots/total_rewards.png")
    plt.close(fig)
    
    print("Creating component rewards plot")
    fig = visualizer.plot_component_rewards()
    fig.savefig("./output/plots/component_rewards.png")
    plt.close(fig)
    
    print("Creating component contribution chart")
    fig = visualizer.component_contribution()
    fig.savefig("./output/plots/component_contribution.png")
    plt.close(fig)
    
    # Save all plots at once
    print("Saving all plots")
    visualizer.save_plots("./output/plots")
    
    print("\n✅ Visualization example completed successfully!")
    print(f"Plots saved to ./output/plots/")

if __name__ == "__main__":
    main()
