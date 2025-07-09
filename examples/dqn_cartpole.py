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
    """Example of using DQN on CartPole with improved hyperparameters."""
    # Create a logger
    logger = Logger(name="dqn_cartpole_fixed", use_wandb=False)
    
    # Create an environment
    env = rllama.make_env(
        "CartPole-v1",
        seed=42,
    )
    
    # Create a DQN agent with better hyperparameters
    agent = rllama.make_agent(
        "DQN",
        env=env,
        learning_rate=0.0005,  # Lower learning rate
        batch_size=128,        # Larger batch
        buffer_size=10000,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.99,    # Slower decay
        target_update_freq=100, # Less frequent updates
        double_dqn=True,
        max_grad_norm=1.0,     # Enable gradient clipping
        reward_scale=0.1,      # Scale rewards for stability
        gamma=0.99,
        verbose=True,
    )
    
    # Create an experiment
    experiment = rllama.Experiment(
        name="dqn_cartpole_fixed",
        agent=agent,
        env=env,
        config={"algorithm": "DQN", "env": "CartPole-v1"},
        logger=logger,
        save_freq=10000,
        eval_freq=2000,
    )
    
    # Train for same number of steps
    experiment.train(
        total_steps=50000,
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