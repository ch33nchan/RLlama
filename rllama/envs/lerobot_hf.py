"""
Wrapper for Hugging Face's LeRobot environments to work with RLlama.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import importlib

# Try to import the real LeRobot or fall back to our stub
try:
    import lerobot
    from lerobot.envs import make as lerobot_make
    from lerobot.envs import registry
    LEROBOT_AVAILABLE = True
    USING_STUB = False
    warnings.warn("Using official LeRobot package")
except ImportError:
    # Fall back to our stub implementation
    warnings.warn(
        "LeRobot package not available (requires Python 3.10+). "
        "Using stub implementation for testing."
    )
    # Import our stub module instead
    from rllama.envs.lerobot_stub import make as lerobot_make
    from rllama.envs.lerobot_stub import registry
    LEROBOT_AVAILABLE = True  # We're using the stub, so technically "available"
    USING_STUB = True

class LeRobotHFWrapper(gym.Wrapper):
    """
    Wrapper for Hugging Face's LeRobot environments to work with RLlama.
    
    This wrapper adapts LeRobot environments to use the Gymnasium API
    expected by RLlama agents.
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
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot is not available. Please install it with: "
                "pip install lerobot (requires Python 3.10+)"
            )
        
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
        
        # Track if we're using the stub
        self.using_stub = USING_STUB
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dictionary
        """
        self.current_step = 0
        return self.env.reset(seed=seed, options=options)
        
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Observation, reward, terminated, truncated, info
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
) -> LeRobotHFWrapper:
    """
    Create a LeRobot environment wrapped for RLlama.
    
    Args:
        env_id: ID of the LeRobot environment
        render_mode: Rendering mode
        max_episode_steps: Maximum steps per episode
        **kwargs: Additional arguments for the environment
        
    Returns:
        Wrapped LeRobot environment
    """
    if not LEROBOT_AVAILABLE:
        raise ImportError(
            "LeRobot is not available. Please install it with: "
            "pip install lerobot (requires Python 3.10+)"
        )
        
    return LeRobotHFWrapper(
        env_id=env_id,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )

# List of available LeRobot environments
def list_lerobot_envs():
    """List all available LeRobot environments."""
    if not LEROBOT_AVAILABLE:
        raise ImportError(
            "LeRobot is not available. Please install it with: "
            "pip install lerobot (requires Python 3.10+)"
        )
    
    try:
        return list(registry.keys())
    except (ImportError, AttributeError) as e:
        print(f"Error accessing environment registry: {e}")
        return []

# Check if we're using the stub or real implementation
def is_using_stub():
    """Check if we're using the stub implementation."""
    return USING_STUB