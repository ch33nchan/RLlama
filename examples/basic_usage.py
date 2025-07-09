import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

import rllama
from rllama.utils.logger import Logger

def main():
    """
    Demonstrates basic usage of the RLlama framework.
    
    This example:
    1. Creates an environment
    2. Initializes an agent
    3. Trains the agent
    4. Evaluates the agent
    5. Plots the learning curve
    """
    # Create environment
    env_id = "CartPole-v1"
    env = rllama.make_env(env_id, seed=42)
    
    # Create agent (DQN for discrete action space)
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
        device="auto",
        verbose=True
    )
    
    # Set up experiment
    experiment = rllama.Experiment(
        agent=agent,
        env=env,
        name=f"DQN-{env_id}",
        log_dir="logs"
    )
    
    # Train the agent
    experiment.train(total_steps=10000, log_interval=1000)
    
    # Evaluate the agent
    mean_reward, std_reward = experiment.evaluate(
        n_episodes=10,
        deterministic=True
    )
    print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot the learning curve
    logger = Logger("logs")
    logger.plot_metrics(
        ["episode_reward"],
        experiment_name=f"DQN-{env_id}",
        smoothing_window=3,
        show=True
    )

if __name__ == "__main__":
    main()