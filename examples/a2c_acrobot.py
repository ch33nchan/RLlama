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
    """Example of using A2C on Acrobot."""
    # Create a logger
    logger = Logger(name="a2c_acrobot", use_wandb=False)
    
    # Create an environment
    env = rllama.make_env(
        "Acrobot-v1",
        seed=42,
        normalize_obs=True,
    )
    
    # Create an agent
    agent = rllama.make_agent(
        "A2C",
        env=env,
        lr=7e-4,
        n_steps=5,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        verbose=True,
    )
    
    # Create an experiment
    experiment = rllama.Experiment(
        name="a2c_acrobot",
        agent=agent,
        env=env,
        config={"algorithm": "A2C", "env": "Acrobot-v1"},
        logger=logger,
        save_freq=20000,
        eval_freq=5000,
    )
    
    # Train the agent
    experiment.train(
        total_steps=200000,
        eval_episodes=5,
        progress_bar=True,
    )
    
    # Evaluate the trained agent
    eval_metrics = experiment.evaluate(episodes=10)
    print(f"Evaluation metrics: {eval_metrics}")
    
    # Save the final model
    final_checkpoint = experiment.save_checkpoint(experiment.agent.total_steps)
    print(f"Saved final model to {final_checkpoint}")
    
    # Close logger
    logger.close()


if __name__ == "__main__":
    main()