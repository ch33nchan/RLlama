import numpy as np
import gymnasium as gym
import torch

import rllama
from rllama.utils.logger import Logger

def main():
    """
    Demonstrates using different rollout lengths with PPO.
    
    PPO naturally uses n-step returns via its rollout length,
    this example compares different rollout lengths.
    """
    # Create environment
    env_id = "LunarLander-v2"
    env = rllama.make_env(env_id, seed=42)
    
    # Create PPO agent with longer rollouts
    agent = rllama.make_agent(
        "PPO",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,  # Long rollouts
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="auto",
        verbose=True
    )
    
    # Set up experiment
    experiment = rllama.Experiment(
        agent=agent,
        env=env,
        name=f"Long-PPO-{env_id}",
        log_dir="logs"
    )
    
    # Train the agent
    experiment.train(total_steps=500000, log_interval=10000)
    
    # Compare with standard PPO
    std_agent = rllama.make_agent(
        "PPO",
        env=env,
        learning_rate=3e-4,
        n_steps=128,  # Standard rollouts
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="auto",
        verbose=True
    )
    
    std_experiment = rllama.Experiment(
        agent=std_agent,
        env=env,
        name=f"Standard-PPO-{env_id}",
        log_dir="logs"
    )
    
    # Train the standard agent
    std_experiment.train(total_steps=500000, log_interval=10000)
    
    # Plot comparison
    logger = Logger("logs")
    logger.plot_comparison(
        ["episode_reward"],
        experiment_names=[f"Long-PPO-{env_id}", f"Standard-PPO-{env_id}"],
        labels=["Long Rollouts (2048)", "Standard Rollouts (128)"],
        smoothing_window=5,
        show=True
    )

if __name__ == "__main__":
    main()