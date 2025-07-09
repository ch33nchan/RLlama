"""
Fixed training script for LeRobot with RLlama.
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

import rllama
from rllama.envs.lerobot_hf import make_lerobot_env, list_lerobot_envs, is_using_stub

class FlattenObservation(gym.ObservationWrapper):
    """
    Wrapper to flatten dictionary observations into a single vector.
    This makes it compatible with standard neural network inputs.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate the total size of the flattened observation
        if isinstance(env.observation_space, spaces.Dict):
            total_size = 0
            for space in env.observation_space.values():
                if isinstance(space, spaces.Box):
                    total_size += np.prod(space.shape)
            
            # Create the new observation space
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
            )
            
            print(f"Created flattened observation space with shape {self.observation_space.shape}")
        else:
            # If not a Dict space, keep the original
            print("Environment already has a non-dictionary observation space")
    
    def observation(self, obs):
        """Convert dict observation to flat vector."""
        if isinstance(self.env.observation_space, spaces.Dict):
            # Flatten the observation dictionary into a single vector
            flattened = []
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    flattened.append(value.flatten())
            
            # Concatenate all flattened arrays
            return np.concatenate(flattened)
        else:
            # If not a Dict space, return the original
            return obs

def main():
    # Check if we're using the stub
    if is_using_stub():
        print("NOTICE: Using stub implementation of LeRobot")
    else:
        print("Using official LeRobot package")
    
    # List available environments
    print("\nAvailable LeRobot environments:")
    envs = list_lerobot_envs()
    for i, env_id in enumerate(envs):
        print(f"{i+1}. {env_id}")
    
    # Create environment
    env_id = "ReachTarget-v0"  # You can change this to any available environment
    print(f"\nCreating environment: {env_id}")
    raw_env = make_lerobot_env(
        env_id=env_id,
        max_episode_steps=100
    )
    
    # Wrap with the flattening wrapper
    env = FlattenObservation(raw_env)
    
    # Print the action and observation spaces
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Create SAC agent with explicit parameters
    print("\nCreating SAC agent with explicit parameters...")
    try:
        agent = rllama.make_agent(
            "SAC",
            env=env,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            batch_size=64,
            buffer_size=10000,
            gamma=0.99,
            tau=0.005,
            device="cpu",
            verbose=True
        )
        
        # Create a config dictionary for the experiment
        config: Dict[str, Any] = {
            "total_timesteps": 10000,
            "eval_episodes": 5,
            "seed": 42,
            "training": {
                "batch_size": 64,
                "buffer_size": 10000,
                "learning_rate": 3e-4,
                "gamma": 0.99
            },
            "env": {
                "id": env_id,
                "max_episode_steps": 100
            }
        }
        
        # Set up experiment with the required config
        print("\nSetting up experiment with config...")
        experiment = rllama.Experiment(
            name=f"SAC-{env_id}",
            agent=agent,
            env=env,
            config=config
        )
        
        # Train for just a few steps to test
        print("\nTraining agent for 1000 steps...")
        # Try different parameter names for the train method
        try:
            experiment.train(total_steps=1000)
        except TypeError:
            try:
                experiment.train(total_timesteps=1000)
            except TypeError:
                try:
                    experiment.train(timesteps=1000)
                except TypeError:
                    print("Could not determine the correct parameter for training length.")
                    print("Please check the train method signature in your RLlama code.")
        
        print("\nTraining completed!")
        
    except Exception as e:
        print(f"Error creating or training agent: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProcess completed!")

if __name__ == "__main__":
    main()