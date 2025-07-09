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
    """Example of using TD3 on Pendulum."""
    # Create a logger
    logger = Logger(name="td3_pendulum", use_wandb=False)
    
    # Create an environment
    env = rllama.make_env(
        "Pendulum-v1",
        seed=42,
        normalize_obs=True,
    )
    
    # Create a TD3 agent
    agent = rllama.make_agent(
        "TD3",
        env=env,
        actor_lr=3e-4,
        critic_lr=3e-4,
        batch_size=100,
        buffer_size=1000000,
        gamma=0.99,
        tau=0.005,          # Soft update parameter
        exploration_noise=0.1,
        policy_noise=0.2,   # Noise added to target actions
        noise_clip=0.5,     # Clipping of noise
        policy_delay=2,     # Frequency of actor updates
        reward_scale=1.0,   # Scale rewards for stability
        max_grad_norm=1.0,
        verbose=True,
    )
    
    # Create an experiment
    experiment = rllama.Experiment(
        name="td3_pendulum",
        agent=agent,
        env=env,
        config={"algorithm": "TD3", "env": "Pendulum-v1"},
        logger=logger,
        save_freq=20000,
        eval_freq=5000,
    )
    
    # Train the agent
    experiment.train(
        total_steps=200000,
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