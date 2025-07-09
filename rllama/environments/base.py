from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class BaseEnvironment(gym.Env, ABC):
    """
    Base class for custom environments.
    
    This class extends Gym's Env class with additional functionality
    specific to RLlama.
    
    Attributes:
        observation_space: Space of possible observations
        action_space: Space of possible actions
        metadata: Environment metadata
    """
    
    def __init__(self):
        """Initialize a base environment."""
        super().__init__()
        
        # These need to be defined by subclasses
        self.observation_space = None
        self.action_space = None
        self.metadata = {"render_modes": []}
        
        # Internal state
        self._state = None
        self._steps = 0
        
    @abstractmethod
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed to use
            options: Additional options for resetting
            
        Returns:
            Initial observation and info dictionary
        """
        if seed is not None:
            super().reset(seed=seed)
            
        self._steps = 0
        
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._steps += 1
        
    @abstractmethod
    def render(self) -> Union[np.ndarray, str, None]:
        """
        Render the environment.
        
        Returns:
            Rendered frame, text representation, or None
        """
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def get_state(self) -> Any:
        """
        Get the current state of the environment.
        
        Returns:
            Current state
        """
        return self._state
    
    def set_state(self, state: Any) -> None:
        """
        Set the current state of the environment.
        
        Args:
            state: State to set
        """
        self._state = state