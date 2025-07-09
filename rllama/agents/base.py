from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rllama.core.experience import Experience


class BaseAgent(ABC):
    """
    Base class for all reinforcement learning agents.
    
    Attributes:
        env: Environment the agent interacts with
        device: Device to run computations on
        gamma: Discount factor
        verbose: Whether to print verbose output
    """
    
    def __init__(
        self,
        env: Any,
        device: str = "auto",
        gamma: float = 0.99,
        verbose: bool = False,
    ):
        """
        Initialize a base agent.
        
        Args:
            env: Environment the agent interacts with
            device: Device to run computations on
            gamma: Discount factor
            verbose: Whether to print verbose output
        """
        self.env = env
        self.gamma = gamma
        self.verbose = verbose
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Get observation and action spaces
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        
        # Initialize step counters
        self.total_steps = 0
        self.episode_steps = 0
        self.episodes = 0
        
    @abstractmethod
    def select_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        evaluate: bool = False
    ) -> Union[int, float, np.ndarray]:
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation
            evaluate: Whether to select deterministically (for evaluation)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Returns:
            Dictionary of metrics from the training step
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the agent's parameters based on a batch of experiences.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of metrics from the update
        """
        pass
    
    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state for saving.
        
        Returns:
            Dictionary containing the agent's state
        """
        pass
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        pass
    
    def collect_experience(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: Union[int, float, np.ndarray], 
        next_obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        reward: float, 
        done: bool, 
        info: Dict[str, Any],
        log_prob: Optional[float] = None,
        value: Optional[float] = None
    ) -> Experience:
        """
        Create an Experience object from a transition.
        
        Args:
            obs: Observation
            action: Action taken
            next_obs: Next observation
            reward: Reward received
            done: Whether the episode is done
            info: Additional information
            log_prob: Log probability of the action (for stochastic policies)
            value: Value estimate (for actor-critic methods)
            
        Returns:
            Experience object
        """
        return Experience(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info,
            log_prob=log_prob,
            value=value
        )
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent to
        """
        state = {
            "agent_state": self.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes
        }
        torch.save(state, path)
        
        if self.verbose:
            print(f"Saved agent to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state["agent_state"])
        self.total_steps = state.get("total_steps", 0)
        self.episodes = state.get("episodes", 0)
        
        if self.verbose:
            print(f"Loaded agent from {path}")