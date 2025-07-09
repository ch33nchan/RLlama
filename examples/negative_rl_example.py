import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

import rllama
from rllama.utils.logger import Logger

def main():
    """
    Demonstrates common pitfalls in reinforcement learning.
    
    This example shows:
    1. Using inappropriate hyperparameters
    2. Using an inappropriate algorithm for the task
    3. Insufficient exploration
    4. Reward design issues
    """
    # Create environment
    env_id = "Acrobot-v1"
    env = rllama.make_env(env_id, seed=42)
    
    # 1. Inappropriate hyperparameters - learning rate too high
    bad_agent1 = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=1.0,  # Too high learning rate
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        device="auto",
        verbose=True
    )
    
    experiment1 = rllama.Experiment(
        agent=bad_agent1,
        env=env,
        name=f"BadLR-DQN-{env_id}",
        log_dir="logs"
    )
    
    # Train the agent with bad learning rate
    experiment1.train(total_steps=50000, log_interval=5000)
    
    # 2. Inappropriate algorithm - using DDPG for discrete action space
    # This will raise an error, but we'll catch it for demonstration
    try:
        bad_agent2 = rllama.make_agent(
            "DDPG",
            env=env,  # Acrobot has discrete action space
            buffer_size=10000,
            batch_size=64,
            gamma=0.99,
            device="auto",
            verbose=True
        )
        
        experiment2 = rllama.Experiment(
            agent=bad_agent2,
            env=env,
            name=f"WrongAlgo-DDPG-{env_id}",
            log_dir="logs"
        )
        
        # This should fail
        experiment2.train(total_steps=50000, log_interval=5000)
    except Exception as e:
        print(f"Error using DDPG with discrete action space: {e}")
        
    # 3. Insufficient exploration
    bad_agent3 = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        exploration_fraction=0.01,  # Very short exploration
        exploration_final_eps=0.01,  # Very little exploration
        device="auto",
        verbose=True
    )
    
    experiment3 = rllama.Experiment(
        agent=bad_agent3,
        env=env,
        name=f"NoExplore-DQN-{env_id}",
        log_dir="logs"
    )
    
    # Train the agent with insufficient exploration
    experiment3.train(total_steps=50000, log_interval=5000)
    
    # Now train a good agent for comparison
    good_agent = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        exploration_fraction=0.2,  # Proper exploration
        exploration_final_eps=0.05,
        device="auto",
        verbose=True
    )
    
    good_experiment = rllama.Experiment(
        agent=good_agent,
        env=env,
        name=f"Good-DQN-{env_id}",
        log_dir="logs"
    )
    
    # Train the good agent
    good_experiment.train(total_steps=50000, log_interval=5000)
    
    # Plot comparison
    logger = Logger("logs")
    logger.plot_comparison(
        ["episode_reward"],
        experiment_names=[
            f"BadLR-DQN-{env_id}", 
            f"NoExplore-DQN-{env_id}", 
            f"Good-DQN-{env_id}"
        ],
        labels=[
            "Bad Learning Rate", 
            "Insufficient Exploration", 
            "Good Configuration"
        ],
        smoothing_window=5,
        show=True
    )
    
    # Print lessons learned
    print("\nCommon RL Pitfalls and Lessons:")
    print("1. Inappropriate Hyperparameters:")
    print("   - Too high learning rates cause instability")
    print("   - Too low learning rates cause slow learning")
    print("   - Always perform hyperparameter tuning")
    
    print("\n2. Algorithm Selection:")
    print("   - Match algorithm to action space (discrete vs continuous)")
    print("   - Consider sample efficiency requirements")
    print("   - Consider exploration-exploitation trade-offs")
    
    print("\n3. Exploration:")
    print("   - Insufficient exploration leads to suboptimal policies")
    print("   - Balance exploration and exploitation")
    print("   - Consider intrinsic motivation for hard exploration tasks")
    
    print("\n4. Reward Design:")
    print("   - Sparse rewards make learning difficult")
    print("   - Reward shaping can help guide learning")
    print("   - Be careful of reward hacking (agent finding unintended behaviors)")

if __name__ == "__main__":
    main()