#!/usr/bin/env python3
"""
Example demonstrating RLlama integration with OpenAI Gym environments.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama import RewardEngine
from rllama.integration import RLlamaGymWrapper

def main():
    """Run gym integration example"""
    print("RLlama Gym Integration Example")
    print("=" * 40)
    
    try:
        # Check if gym is available
        import gym
    except ImportError:
        print("Error: OpenAI Gym is required for this example.")
        print("Please install with: pip install gym")
        return
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    print("\n1. Setting up reward configuration")
    print("-" * 40)
    
    # Create a reward config for CartPole
    config = {
        "reward_components": [
            {
                "name": "LengthReward",
                "params": {
                    "target_length": 200,
                    "strength": 0.001
                }
            },
            {
                "name": "ConstantReward",
                "params": {
                    "value": 0.1
                }
            }
        ],
        "shaping_config": {
            "LengthReward": {"weight": 0.5},
            "ConstantReward": {"weight": 1.0}
        },
        "logging": {
            "log_dir": "./output",
            "log_frequency": 1
        }
    }
    
    # Save config
    config_path = "./output/cartpole_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Created reward config: {config_path}")
    
    # Create CartPole environment
    print("\n2. Setting up environment")
    print("-" * 40)
    
    env = gym.make("CartPole-v0")
    print(f"Created environment: {env.spec.id}")
    
    # Create RLlama reward engine
    engine = RewardEngine(config_path)
    
    # Custom context mapping function
    def cartpole_context_mapper(obs, action, reward, done, info):
        return {
            "observation": obs,
            "action": action,
            "env_reward": reward,
            "done": done,
            "info": info,
            "cart_position": obs[0],
            "cart_velocity": obs[1],
            "pole_angle": obs[2],
            "pole_velocity": obs[3]
        }
    
    # Wrap environment with RLlama
    wrapped_env = RLlamaGymWrapper(
        env=env,
        reward_engine=engine,
        context_mapper=cartpole_context_mapper,
        combine_rewards=True
    )
    
    print("Created RLlama wrapped environment")
    
    # Run episodes to compare rewards
    print("\n3. Running episodes")
    print("-" * 40)
    
    num_episodes = 10
    
    # Track rewards
    original_rewards = []
    shaped_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        episode_original_reward = 0
        episode_shaped_reward = 0
        episode_length = 0
        
        obs = wrapped_env.reset()
        done = False
        
        while not done:
            # Take random action
            action = wrapped_env.action_space.sample()
            
            # Step environment
            obs, reward, done, info = wrapped_env.step(action)
            
            # Track rewards
            episode_original_reward += info['original_reward']
            episode_shaped_reward += reward
            episode_length += 1
            
        # Save episode results
        original_rewards.append(episode_original_reward)
        shaped_rewards.append(episode_shaped_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode+1}: Length={episode_length}, Original Reward={episode_original_reward}, Shaped Reward={episode_shaped_reward:.2f}")
    
    wrapped_env.close()
    
    # Plot results
    print("\n4. Visualizing results")
    print("-" * 40)
    
    # Plot episode lengths
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_episodes+1), episode_lengths, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Episode Lengths')
    plt.grid(True, alpha=0.3)
    
    # Plot rewards
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_episodes+1), original_rewards, marker='o', label='Original')
    plt.plot(range(1, num_episodes+1), shaped_rewards, marker='s', label='Shaped')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Original vs. Shaped Rewards')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("./output/gym_integration_results.png")
    
    print(f"Saved results to ./output/gym_integration_results.png")
    print("\n✅ Gym integration example completed successfully!")

if __name__ == "__main__":
    main()
