from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np

from rllama.environments.base import BaseEnvironment
from rllama.environments.wrappers import (
    NormalizeObservation,
    NormalizeReward,
    TimeLimit,
    FrameStack,
    VecEnv
)

# Registry of custom environments
ENV_REGISTRY: Dict[str, Type[BaseEnvironment]] = {}


def make_env(
    env_id: str,
    seed: Optional[int] = None,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
    clip_obs: Optional[float] = 10.0,
    clip_reward: Optional[float] = 10.0,
    time_limit: Optional[int] = None,
    frame_stack: Optional[int] = None,
    vec_env_size: Optional[int] = None,
    **kwargs
) -> gym.Env:
    """
    Create a Gym environment with optional wrappers.
    
    Args:
        env_id: ID of the environment to create
        seed: Random seed
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        clip_obs: Maximum observation values after normalization
        clip_reward: Maximum reward values after normalization
        time_limit: Maximum number of steps per episode
        frame_stack: Number of frames to stack
        vec_env_size: Number of environments to run in parallel
        **kwargs: Additional arguments to pass to the environment constructor
        
    Returns:
        Created environment
        
    Raises:
        ValueError: If the environment ID is not valid
    """
    # Try to create a custom environment
    if env_id in ENV_REGISTRY:
        env = ENV_REGISTRY[env_id](**kwargs)
    else:
        # Fall back to Gym
        try:
            env = gym.make(env_id, **kwargs)
        except gym.error.NameNotFound:
            raise ValueError(f"Environment '{env_id}' not found in Gym or custom registry")
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
        
    # Apply wrappers
    if normalize_obs:
        env = NormalizeObservation(env, clip=clip_obs)
        
    if normalize_reward:
        env = NormalizeReward(env, clip=clip_reward)
        
    if time_limit is not None:
        env = TimeLimit(env, max_steps=time_limit)
        
    if frame_stack is not None and frame_stack > 1:
        env = FrameStack(env, num_stack=frame_stack)
        
    # Create vectorized environment if requested
    if vec_env_size is not None and vec_env_size > 1:
        # Create multiple instances of the environment
        def make_single_env():
            env_instance = make_env(
                env_id=env_id,
                seed=seed,
                normalize_obs=normalize_obs,
                normalize_reward=normalize_reward,
                clip_obs=clip_obs,
                clip_reward=clip_reward,
                time_limit=time_limit,
                frame_stack=frame_stack,
                vec_env_size=None,
                **kwargs
            )
            if seed is not None:
                seed += 1
            return env_instance
            
        envs = [make_single_env() for _ in range(vec_env_size)]
        env = VecEnv(envs)
        
    return env


def register_env(name: str, env_class: Type[BaseEnvironment]) -> None:
    """
    Register a custom environment.
    
    Args:
        name: Name to register the environment under
        env_class: Environment class to register
        
    Raises:
        ValueError: If an environment with the same name is already registered
    """
    if name in ENV_REGISTRY:
        raise ValueError(f"Environment '{name}' is already registered")
    
    ENV_REGISTRY[name] = env_class