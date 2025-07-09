import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

def make_sb3_env(
    env_id: str,
    n_envs: int = 1,
    seed: Optional[int] = None,
    vec_env_cls: Optional[Any] = None,
    monitor_dir: Optional[str] = None,
    normalize: bool = False,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
) -> Any:
    """
    Create a vectorized environment compatible with Stable Baselines 3.
    
    Args:
        env_id: The environment ID
        n_envs: Number of environments to run in parallel
        seed: Random seed
        vec_env_cls: Vectorized environment class to use
        monitor_dir: Directory to save Monitor logs
        normalize: Whether to normalize observations and rewards
        norm_obs: Whether to normalize observations
        norm_reward: Whether to normalize rewards
        clip_obs: Maximum absolute value for observations
        clip_reward: Maximum absolute value for rewards
        gamma: Discount factor for reward normalization
        
    Returns:
        Vectorized environment
    """
    if not SB3_AVAILABLE:
        raise ImportError(
            "Stable Baselines 3 is not available. Please install it with: "
            "pip install stable-baselines3"
        )
    
    # Define environment creation function
    def make_env(env_id, rank, seed=None):
        def _init():
            env = gym.make(env_id)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            
            # Wrap with Monitor if directory is provided
            if monitor_dir:
                import os
                os.makedirs(monitor_dir, exist_ok=True)
                env = Monitor(env, f"{monitor_dir}/env_{rank}")
                
            return env
        return _init
    
    # Create environment functions
    env_fns = [make_env(env_id, i, seed) for i in range(n_envs)]
    
    # Choose vectorization method
    if vec_env_cls is None:
        vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        
    # Create vectorized environment
    env = vec_env_cls(env_fns)
    
    # Apply normalization if requested
    if normalize:
        env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
        )
        
    return env

class SB3RLlamaWrapper:
    """
    Wrapper to integrate SB3 vectorized environments with RLlama.
    
    This wrapper adapts the SB3 VecEnv interface to the Gymnasium Env
    interface that RLlama expects.
    """
    
    def __init__(
        self,
        vec_env: Any,
        single_action_space: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize the wrapper.
        
        Args:
            vec_env: SB3 vectorized environment
            single_action_space: Whether to return a single action space
            device: Device to use for tensor operations
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "Stable Baselines 3 is not available. Please install it with: "
                "pip install stable-baselines3"
            )
            
        self.vec_env = vec_env
        self.single_action_space = single_action_space
        self.device = device
        
        # Get observation and action spaces
        self.observation_space = vec_env.observation_space
        if single_action_space:
            self.action_space = vec_env.action_space
        else:
            self.action_space = vec_env.action_space.spaces[0]
            
        # Track environment state
        self.current_obs = None
        
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            Observation and info dictionary
        """
        obs = self.vec_env.reset(**kwargs)
        
        # Handle different return formats
        if isinstance(obs, tuple):
            self.current_obs = obs[0]
            return obs[0], obs[1]
        else:
            self.current_obs = obs
            return obs, {}
            
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Convert to batched action if needed
        if self.single_action_space:
            action = [action]
            
        # Take a step in the vectorized environment
        obs, rewards, dones, infos = self.vec_env.step(action)
        
        # Convert to the Gymnasium step API format
        if isinstance(infos, dict):
            # SB3 sometimes returns a dict instead of a list of dicts
            truncated = infos.get("TimeLimit.truncated", False)
            infos = [infos]
        else:
            # Check for truncation in the first environment
            truncated = infos[0].get("TimeLimit.truncated", False) if infos else False
            
        # Update current observation
        self.current_obs = obs
        
        # Return results for the first environment
        return obs[0], rewards[0], dones[0], truncated, infos[0]
        
    def close(self):
        """Close the environment."""
        self.vec_env.close()
        
    def render(self, mode="human"):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendering output
        """
        return self.vec_env.render(mode=mode)