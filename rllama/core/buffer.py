from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import random

from rllama.core.experience import Experience


class Buffer:
    """
    Base buffer class for storing experiences.
    
    Attributes:
        capacity: Maximum number of experiences to store
        device: Device to store experiences on
    """
    
    def __init__(self, capacity: int, device: str = "cpu"):
        """
        Initialize a buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store experiences on
        """
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
        
    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience to add
        """
        raise NotImplementedError
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of tensors
        """
        raise NotImplementedError
        
    def __len__(self) -> int:
        """
        Get the number of experiences in the buffer.
        
        Returns:
            Number of experiences
        """
        return len(self.buffer)
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
        self.position = 0


class ReplayBuffer(Buffer):
    """
    Replay buffer for off-policy algorithms.
    
    Stores experiences and samples them randomly.
    """
    
    def __init__(self, capacity: int, device: str = "cpu"):
        """
        Initialize a replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store experiences on
        """
        super().__init__(capacity, device)
        self.buffer = []
        self.position = 0
        
    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience to add
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of tensors
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Convert to tensors
        from rllama.core.experience import Experience
        batch = Experience.to_tensor_batch(experiences, device=self.device)
        
        return batch


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized replay buffer for off-policy algorithms.
    
    Samples experiences with higher TD error more frequently.
    """
    
    def __init__(
        self, 
        capacity: int, 
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu"
    ):
        """
        Initialize a prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction (0 = no correction, 1 = full correction)
            beta_increment_per_sampling: Increment beta by this amount each time sample is called
            epsilon: Small constant added to priority to ensure non-zero probabilities
            device: Device to store experiences on
        """
        super().__init__(capacity, device)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        self.max_priority = 1.0
        
    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience to add
        """
        # Get index for new experience
        idx = self.position
        
        # Add experience
        super().add(experience)
        
        # Set priority to maximum priority
        self.priorities[idx] = self.max_priority
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of tensors
        """
        batch_size = min(batch_size, len(self.buffer))
        
        # Compute sampling probabilities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Convert priorities to probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(
            len(probabilities), 
            batch_size, 
            replace=False, 
            p=probabilities
        )
        
        # Compute importance sampling weights
        weights = (len(probabilities) * probabilities[indices]) ** -self.beta
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Convert to tensors
        from rllama.core.experience import Experience
        batch = Experience.to_tensor_batch(experiences, device=self.device)
        
        # Add indices and weights to batch
        batch["indices"] = torch.tensor(indices, device=self.device)
        batch["weights"] = weights
        
        return batch
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            # Add small constant to ensure non-zero probability
            self.priorities[idx] = priority + self.epsilon
            
            # Update maximum priority
            self.max_priority = max(self.max_priority, self.priorities[idx])


class EpisodeBuffer(Buffer):
    """
    Episode buffer for on-policy algorithms.
    
    Stores all experiences in the order they are added.
    """
    
    def __init__(self, capacity: int, device: str = "cpu"):
        """
        Initialize an episode buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store experiences on
        """
        super().__init__(capacity, device)
        self.buffer = []
        
    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            experience: Experience to add
        """
        self.buffer.append(experience)
        
        # Remove oldest experiences if over capacity
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of tensors
        """
        # Make sure batch_size doesn't exceed buffer size
        batch_size = min(batch_size, len(self.buffer))
        
        # Sample random indices
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]
        
        # Convert to tensors using the Experience static method
        from rllama.core.experience import Experience
        batch = Experience.to_tensor_batch(experiences, device=self.device)
        
        return batch
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Dictionary of tensors
        """
        return self.sample_batch(batch_size)
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        Get all experiences in the buffer.
        
        Returns:
            Dictionary of tensors
        """
        # Convert to tensors
        from rllama.core.experience import Experience
        batch = Experience.to_tensor_batch(self.buffer, device=self.device)
        
        return batch