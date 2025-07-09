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
    """Example of using MBPO on Pendulum."""
    # Create a logger
    logger = Logger(name="mbpo_pendulum", use_wandb=False)
    
    # Create an environment
    env = rllama.make_env(
        "Pendulum-v1",
        seed=42,
        normalize_obs=True,
    )
    
    # Create an MBPO agent
    agent = rllama.make_agent(
        "MBPO",
        env=env,
        real_batch_size=256,
        model_batch_size=256,
        real_buffer_size=100000,
        model_buffer_size=200000,
        model_ensemble_size=5,
        model_hidden_dims=[200, 200, 200, 200],
        model_learning_rate=1e-3,
        horizon=1,                # Planning horizon
        real_ratio=0.05,          # Ratio of real data in training
        updates_per_step=20,      # Policy updates per step
        model_updates_per_step=40,  # Model updates per step
        model_rollout_batch_size=50000,
        model_retain_epochs=1,
        model_train_frequency=250,
        num_model_rollouts=400,
        gamma=0.99,
        verbose=True,
    )
    
    # Create an experiment
    experiment = rllama.Experiment(
        name="mbpo_pendulum",
        agent=agent,
        env=env,
        config={
            "algorithm": "MBPO", 
            "env": "Pendulum-v1",
            "horizon": 1,
            "real_ratio": 0.05,
        },
        logger=logger,
        save_freq=20000,
        eval_freq=5000,
    )
    
    # Train the agent
    experiment.train(
        total_steps=500,  # MBPO needs fewer real environment steps
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