import numpy as np
import gymnasium as gym
import torch

import rllama
from rllama.utils.logger import Logger

def main():
    """
    Demonstrates using an n-step DQN agent.
    
    N-step DQN uses n-step returns for Q-learning updates,
    which can speed up learning by propagating rewards faster.
    """
    # Create environment
    env_id = "CartPole-v1"
    env = rllama.make_env(env_id, seed=42)
    
    # Create n-step DQN agent
    agent = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        n_step=3,  # Use 3-step returns
        device="auto",
        verbose=True
    )
    
    # Set up experiment
    experiment = rllama.Experiment(
        agent=agent,
        env=env,
        name=f"N-DQN-{env_id}",
        log_dir="logs"
    )
    
    # Train the agent
    experiment.train(total_steps=10000, log_interval=1000)
    
    # Compare with standard DQN
    std_agent = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        target_update_freq=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        n_step=1,  # Standard 1-step returns
        device="auto",
        verbose=True
    )
    
    std_experiment = rllama.Experiment(
        agent=std_agent,
        env=env,
        name=f"DQN-{env_id}",
        log_dir="logs"
    )
    
    # Train the standard agent
    std_experiment.train(total_steps=10000, log_interval=1000)
    
    # Plot comparison
    logger = Logger("logs")
    logger.plot_comparison(
        ["episode_reward"],
        experiment_names=[f"N-DQN-{env_id}", f"DQN-{env_id}"],
        labels=["3-step DQN", "Standard DQN"],
        smoothing_window=3,
        show=True
    )

if __name__ == "__main__":
    main()