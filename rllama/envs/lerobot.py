"""
Wrapper for LeRobot environments to work with RLlama.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

# Try to import the real LeRobot
try:
    import lerobot
    from lerobot.envs import make as lerobot_make
    from lerobot.envs import registry
    LEROBOT_AVAILABLE = True
    print("Using official LeRobot package")
except ImportError:
    raise ImportError(
        "LeRobot is not available. Please install it with: "
        "pip install lerobot (requires Python 3.10+)"
    )

class LeRobotWrapper(gym.Wrapper):
    """
    Wrapper for LeRobot environments to work with RLlama.
    """
    
    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        kwargs: Optional[Dict] = None
    ):
        """
        Initialize the LeRobot wrapper.
        
        Args:
            env_id: ID of the LeRobot environment to use
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            max_episode_steps: Maximum steps per episode
            kwargs: Additional keyword arguments for the environment
        """
        # Create the LeRobot environment
        kwargs = kwargs or {}
        if render_mode:
            kwargs["render_mode"] = render_mode
            
        self.env_id = env_id
        env = lerobot_make(env_id, **kwargs)
        
        # Initialize the wrapper
        super().__init__(env)
        
        # Set episode limit if provided
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Reset the environment.
        """
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)
        
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        """
        self.current_step += 1
        
        # Take action in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check for episode timeout
        if self.max_episode_steps and self.current_step >= self.max_episode_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info

def make_lerobot_env(
    env_id: str,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    **kwargs
) -> LeRobotWrapper:
    """
    Create a LeRobot environment wrapped for RLlama.
    """
    return LeRobotWrapper(
        env_id=env_id,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )

# List of available LeRobot environments
def list_lerobot_envs():
    """List all available LeRobot environments."""
    try:
        return list(registry.keys())
    except (ImportError, AttributeError) as e:
        print(f"Error accessing environment registry: {e}")
        return []