from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

from rllama.utils.logger import Logger


class Callback(ABC):
    """
    Base class for callbacks.
    
    Callbacks can be used to monitor, modify behavior, or log information
    during training.
    """
    
    def on_training_start(self, **kwargs) -> None:
        """Called at the start of training."""
        pass
    
    def on_training_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    def on_episode_start(self, **kwargs) -> None:
        """Called at the start of an episode."""
        pass
    
    def on_episode_end(self, **kwargs) -> None:
        """Called at the end of an episode."""
        pass
    
    def on_step(self, **kwargs) -> None:
        """Called at every step."""
        pass
    
    def on_update(self, **kwargs) -> None:
        """Called after every agent update."""
        pass


class CallbackList:
    """
    Container for managing multiple callbacks.
    
    Attributes:
        callbacks: List of callbacks
    """
    
    def __init__(self, callbacks: List[Callback]):
        """
        Initialize a callback list.
        
        Args:
            callbacks: List of callbacks
        """
        self.callbacks = callbacks
        
    def on_training_start(self, **kwargs) -> None:
        """Called at the start of training."""
        for callback in self.callbacks:
            callback.on_training_start(**kwargs)
            
    def on_training_end(self, **kwargs) -> None:
        """Called at the end of training."""
        for callback in self.callbacks:
            callback.on_training_end(**kwargs)
            
    def on_episode_start(self, **kwargs) -> None:
        """Called at the start of an episode."""
        for callback in self.callbacks:
            callback.on_episode_start(**kwargs)
            
    def on_episode_end(self, **kwargs) -> None:
        """Called at the end of an episode."""
        for callback in self.callbacks:
            callback.on_episode_end(**kwargs)
            
    def on_step(self, **kwargs) -> None:
        """Called at every step."""
        for callback in self.callbacks:
            callback.on_step(**kwargs)
            
    def on_update(self, **kwargs) -> None:
        """Called after every agent update."""
        for callback in self.callbacks:
            callback.on_update(**kwargs)


class LoggingCallback(Callback):
    """
    Callback for logging metrics during training.
    
    Attributes:
        logger: Logger for recording metrics
        log_freq: Frequency of logging (in steps)
    """
    
    def __init__(
        self,
        logger: Optional[Logger] = None,
        log_freq: int = 1,
    ):
        """
        Initialize a logging callback.
        
        Args:
            logger: Logger for recording metrics
            log_freq: Frequency of logging (in steps)
        """
        self.logger = logger or Logger()
        self.log_freq = log_freq
        self.step_count = 0
        
    def on_training_start(self, **kwargs) -> None:
        """Log training start."""
        self.logger.info("Training started")
        
    def on_training_end(self, **kwargs) -> None:
        """Log training end."""
        self.logger.info("Training ended")
        
    def on_episode_start(self, **kwargs) -> None:
        """Log episode start."""
        self.episode_start_time = kwargs.get("time", 0)
        
    def on_episode_end(self, **kwargs) -> None:
        """Log episode end."""
        episode = kwargs.get("episode", 0)
        reward = kwargs.get("reward", 0)
        length = kwargs.get("length", 0)
        
        self.logger.info(f"Episode {episode} ended with reward {reward:.2f} and length {length}")
        self.logger.record("episode_reward", reward)
        self.logger.record("episode_length", length)
        
    def on_step(self, **kwargs) -> None:
        """Log step metrics."""
        self.step_count += 1
        
        if self.step_count % self.log_freq == 0:
            # Log step metrics
            for key, value in kwargs.items():
                if key not in ["time", "agent", "env"]:
                    self.logger.record(key, value)
                    
            # Flush logs
            self.logger.flush()
            
    def on_update(self, **kwargs) -> None:
        """Log update metrics."""
        for key, value in kwargs.items():
            if key not in ["time", "agent", "env"]:
                self.logger.record(f"update/{key}", value)


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on reward threshold.
    
    Attributes:
        reward_threshold: Reward threshold for early stopping
        window_size: Number of episodes to average reward over
        verbose: Whether to print verbose output
    """
    
    def __init__(
        self,
        reward_threshold: float,
        window_size: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize an early stopping callback.
        
        Args:
            reward_threshold: Reward threshold for early stopping
            window_size: Number of episodes to average reward over
            verbose: Whether to print verbose output
        """
        self.reward_threshold = reward_threshold
        self.window_size = window_size
        self.verbose = verbose
        self.rewards = []
        self.stop_training = False
        
    def on_episode_end(self, **kwargs) -> None:
        """Check if training should be stopped."""
        reward = kwargs.get("reward", 0)
        self.rewards.append(reward)
        
        # Only keep the last window_size rewards
        if len(self.rewards) > self.window_size:
            self.rewards.pop(0)
            
        # Check if average reward exceeds threshold
        if len(self.rewards) == self.window_size:
            avg_reward = np.mean(self.rewards)
            
            if avg_reward >= self.reward_threshold:
                self.stop_training = True
                
                if self.verbose:
                    print(f"Early stopping: Reward threshold {self.reward_threshold} reached with average reward {avg_reward:.2f}")
                    
        # Notify the training loop to stop
        kwargs["stop_training"] = self.stop_training


class CheckpointCallback(Callback):
    """
    Callback for saving agent checkpoints.
    
    Attributes:
        save_freq: Frequency of saving checkpoints (in steps)
        save_path: Directory to save checkpoints
        name_prefix: Prefix for checkpoint filenames
        verbose: Whether to print verbose output
    """
    
    def __init__(
        self,
        save_freq: int,
        save_path: str = "./checkpoints",
        name_prefix: str = "agent",
        verbose: bool = True,
    ):
        """
        Initialize a checkpoint callback.
        
        Args:
            save_freq: Frequency of saving checkpoints (in steps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint filenames
            verbose: Whether to print verbose output
        """
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.verbose = verbose
        self.step_count = 0
        
        # Create save directory if it doesn't exist
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    def on_step(self, **kwargs) -> None:
        """Save checkpoint if save_freq steps have passed."""
        self.step_count += 1
        
        if self.step_count % self.save_freq == 0:
            agent = kwargs.get("agent")
            
            if agent is not None:
                # Save agent
                path = f"{self.save_path}/{self.name_prefix}_{self.step_count}.pt"
                agent.save(path)
                
                if self.verbose:
                    print(f"Saved checkpoint to {path}")