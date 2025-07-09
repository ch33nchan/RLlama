import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import torch

import rllama
from rllama.utils.logger import Logger


def main():
    """Example of using DQN on LunarLander."""
    # Create a logger
    logger = Logger(name="dqn_lunarlander", use_wandb=False)
    
    # Create an environment - UPDATED to use v3 instead of v2
    env = rllama.make_env(
        "LunarLander-v3",  # Changed from v2 to v3
        seed=42,
        normalize_obs=True,
    )
    
    # Create an agent
    agent = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=100000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=10,
        double_dqn=True,
        max_grad_norm=10.0,
        verbose=True,
    )
    
    # Create an experiment
    experiment = rllama.Experiment(
        name="dqn_lunarlander",
        agent=agent,
        env=env,
        config={"algorithm": "DQN", "env": "LunarLander-v3"},  # Updated here too
        logger=logger,
        save_freq=50000,
        eval_freq=10000,
    )
    
    # Train the agent
    experiment.train(
        total_steps=500000,
        eval_episodes=10,
        progress_bar=True,
    )
    
    # Evaluate the trained agent
    eval_metrics = experiment.evaluate(episodes=20)
    print(f"Evaluation metrics: {eval_metrics}")
    
    # Save the final model
    final_checkpoint = experiment.save_checkpoint(experiment.agent.total_steps)
    print(f"Saved final model to {final_checkpoint}")
    
    # Close logger
    logger.close()


if __name__ == "__main__":
    main()