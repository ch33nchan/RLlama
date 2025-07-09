from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class Policy(ABC):
    """
    Abstract base class for all policies.
    
    A policy defines how an agent selects actions based on observations.
    """
    
    @abstractmethod
    def select_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False
    ) -> Union[int, float, np.ndarray]:
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation
            deterministic: Whether to select deterministically
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the policy based on a batch of experiences.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of metrics from the update
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the policy to a file.
        
        Args:
            path: Path to save the policy to
        """
        if hasattr(self, "network") and isinstance(self.network, nn.Module):
            torch.save(self.network.state_dict(), path)
        else:
            raise NotImplementedError("Policy does not have a network to save")
            
    def load(self, path: str, device: str = "cpu") -> None:
        """
        Load the policy from a file.
        
        Args:
            path: Path to load the policy from
            device: Device to load the policy to
        """
        if hasattr(self, "network") and isinstance(self.network, nn.Module):
            self.network.load_state_dict(torch.load(path, map_location=device))
        else:
            raise NotImplementedError("Policy does not have a network to load")


class StochasticPolicy(Policy):
    """
    Base class for stochastic policies.
    
    A stochastic policy samples actions from a probability distribution.
    """
    
    def __init__(self, network: nn.Module, device: str = "cpu"):
        """
        Initialize a stochastic policy.
        
        Args:
            network: Neural network to use for the policy
            device: Device to run the policy on
        """
        self.network = network
        self.device = device
        
    def select_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = False
    ) -> Union[int, float, np.ndarray]:
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation
            deterministic: Whether to select deterministically
            
        Returns:
            Selected action
        """
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        elif isinstance(obs, dict):
            obs_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
            }
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
            
        with torch.no_grad():
            if deterministic:
                # Select most likely action
                action = self._get_deterministic_action(obs_tensor)
            else:
                # Sample from distribution
                action, _ = self._get_action_and_log_prob(obs_tensor)
                
        # Convert to numpy and remove batch dimension
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()[0]
            
        return action
    
    @abstractmethod
    def _get_action_and_log_prob(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute its log probability.
        
        Args:
            obs: Current observation
            
        Returns:
            Tuple of (action, log_prob)
        """
        pass
    
    @abstractmethod
    def _get_deterministic_action(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Get the most likely action.
        
        Args:
            obs: Current observation
            
        Returns:
            Most likely action
        """
        pass


class DeterministicPolicy(Policy):
    """
    Base class for deterministic policies.
    
    A deterministic policy selects actions without randomness.
    """
    
    def __init__(self, network: nn.Module, device: str = "cpu"):
        """
        Initialize a deterministic policy.
        
        Args:
            network: Neural network to use for the policy
            device: Device to run the policy on
        """
        self.network = network
        self.device = device
        
    def select_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        deterministic: bool = True
    ) -> Union[int, float, np.ndarray]:
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation
            deterministic: Whether to select deterministically (ignored for deterministic policies)
            
        Returns:
            Selected action
        """
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        elif isinstance(obs, dict):
            obs_tensor = {
                k: torch.FloatTensor(v).unsqueeze(0).to(self.device)
                for k, v in obs.items()
            }
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
            
        with torch.no_grad():
            action = self._get_action(obs_tensor)
                
        # Convert to numpy and remove batch dimension
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()[0]
            
        return action
    
    @abstractmethod
    def _get_action(
        self, 
        obs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Get the action for the given observation.
        
        Args:
            obs: Current observation
            
        Returns:
            Action
        """
        pass