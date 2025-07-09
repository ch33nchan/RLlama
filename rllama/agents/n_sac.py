import numpy as np
import gymnasium as gym
import torch

import rllama
from rllama.utils.logger import Logger

def main():
    """
    Demonstrates using SAC with different return lengths.
    
    This example compares standard SAC (1-step returns)
    with an n-step SAC variant (using multi-step returns).
    """
    # Create environment
    env_id = "Pendulum-v1"
    env = rllama.make_env(env_id, seed=42)
    
    # Create n-step SAC agent
    agent = rllama.make_agent(
        "SAC",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        device="auto",
        verbose=True
    )
    
    # Set up experiment
    experiment = rllama.Experiment(
        agent=agent,
        env=env,
        name=f"N-SAC-{env_id}",
        log_dir="logs"
    )
    
    # Train the agent
    experiment.train(total_steps=100000, log_interval=5000)
    
    # Compare with standard SAC
    std_agent = rllama.make_agent(
        "SAC",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        device="auto",
        verbose=True
    )
    
    std_experiment = rllama.Experiment(
        agent=std_agent,
        env=env,
        name=f"SAC-{env_id}",
        log_dir="logs"
    )
    
    # Train the standard agent
    std_experiment.train(total_steps=100000, log_interval=5000)
    
    # Plot comparison
    logger = Logger("logs")
    logger.plot_comparison(
        ["episode_reward"],
        experiment_names=[f"N-SAC-{env_id}", f"SAC-{env_id}"],
        labels=["N-step SAC", "Standard SAC"],
        smoothing_window=5,
        show=True
    )

if __name__ == "__main__":
    main()