from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class Experience:
    """
    Container for a single step of agent-environment interaction.
    
    Attributes:
        obs: Observation
        action: Action
        reward: Reward
        next_obs: Next observation
        done: Whether the episode is done
        info: Additional information
    """
    
    def __init__(
        self,
        obs: np.ndarray,
        action: Union[int, float, np.ndarray],
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        info: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Initialize an experience.
        
        Args:
            obs: Observation
            action: Action
            next_obs: Next observation
            reward: Reward
            done: Whether the episode is done
            info: Additional information
            **kwargs: Additional data to store
        """
        self.obs = obs
        self.action = action
        self.next_obs = next_obs
        self.reward = reward
        self.done = done
        self.info = info if info is not None else {}
        
        # Store additional data
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experience to a dictionary.
        
        Returns:
            Dictionary containing the experience data
        """
        # Get all attributes
        result = self.__dict__.copy()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """
        Create an experience from a dictionary.
        
        Args:
            data: Dictionary containing the experience data
            
        Returns:
            Created experience
        """
        return cls(**data)
    
    @staticmethod
    def to_tensor(
        experience: 'Experience',
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Convert an experience to tensors.
        
        Args:
            experience: Experience to convert
            device: Device to put tensors on
            
        Returns:
            Dictionary of tensors
        """
        # Convert to dictionary
        data = experience.to_dict()
        
        # Convert each value to a tensor
        result = {}
        for key, value in data.items():
            if key == "info":
                # Skip info dictionary
                continue
                
            if isinstance(value, np.ndarray):
                result[key] = torch.FloatTensor(value).to(device)
            elif isinstance(value, (int, float, bool)):
                result[key] = torch.tensor(value, device=device)
            elif isinstance(value, torch.Tensor):
                result[key] = value.to(device)
                
        return result
    
    @staticmethod
    def to_tensor_batch(
        experiences: List['Experience'],
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a list of experiences to tensors.
        
        Args:
            experiences: Experiences to convert
            device: Device to put tensors on
            
        Returns:
            Dictionary of tensors
        """
        # Get all keys
        keys = set()
        for exp in experiences:
            keys.update(exp.to_dict().keys())
            
        # Remove info
        if "info" in keys:
            keys.remove("info")
            
        # Initialize result
        result = {key: [] for key in keys}
        
        # Collect values
        for exp in experiences:
            exp_dict = exp.to_dict()
            for key in keys:
                if key in exp_dict:
                    result[key].append(exp_dict[key])
                    
        # Convert to tensors
        for key in keys:
            if len(result[key]) == 0:
                continue
                
            if isinstance(result[key][0], np.ndarray):
                result[key] = torch.FloatTensor(np.array(result[key])).to(device)
            elif isinstance(result[key][0], (int, float, bool)):
                result[key] = torch.tensor(result[key], device=device)
            elif isinstance(result[key][0], torch.Tensor):
                result[key] = torch.stack(result[key]).to(device)
                
        return result