from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.Wrapper):
    """
    Normalize observations to have zero mean and unit variance.
    
    Attributes:
        mean: Running mean of observations
        var: Running variance of observations
        count: Number of observations seen
        clip: Maximum observation value after normalization
    """
    
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, clip: Optional[float] = 10.0):
        """
        Initialize a normalize observation wrapper.
        
        Args:
            env: Environment to wrap
            epsilon: Small constant to avoid division by zero
            clip: Maximum observation value after normalization
        """
        super().__init__(env)
        self.mean = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.var = np.ones(self.observation_space.shape, dtype=np.float32)
        self.count = 0
        self.epsilon = epsilon
        self.clip = clip
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and return normalized observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.update_stats(obs)
        return self.normalize(obs), info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step and normalize the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.update_stats(obs)
        return self.normalize(obs), reward, terminated, truncated, info
    
    def update_stats(self, obs: np.ndarray) -> None:
        """Update running statistics."""
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.var += delta * delta2
        
    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize an observation."""
        std = np.sqrt(self.var / (self.count + self.epsilon))
        normalized = (obs - self.mean) / (std + self.epsilon)
        
        if self.clip is not None:
            normalized = np.clip(normalized, -self.clip, self.clip)
            
        return normalized


class NormalizeReward(gym.Wrapper):
    """
    Normalize rewards to have zero mean and unit variance.
    
    Attributes:
        mean: Running mean of rewards
        var: Running variance of rewards
        count: Number of rewards seen
        clip: Maximum reward value after normalization
        return_sum: Sum of returns
        return_count: Number of episodes
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        gamma: float = 0.99, 
        epsilon: float = 1e-8, 
        clip: Optional[float] = 10.0,
        normalize_returns: bool = True
    ):
        """
        Initialize a normalize reward wrapper.
        
        Args:
            env: Environment to wrap
            gamma: Discount factor
            epsilon: Small constant to avoid division by zero
            clip: Maximum reward value after normalization
            normalize_returns: Whether to normalize returns or rewards
        """
        super().__init__(env)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip
        self.normalize_returns = normalize_returns
        
        # Return normalization
        self.return_sum = 0.0
        self.return_count = 0
        self.returns = None
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and return the observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.returns = 0.0
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step and normalize the reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Update reward statistics
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2
        
        # Normalize reward
        std = np.sqrt(self.var / (self.count + self.epsilon))
        norm_reward = reward
        
        if not self.normalize_returns:
            norm_reward = (reward - self.mean) / (std + self.epsilon)
            if self.clip is not None:
                norm_reward = np.clip(norm_reward, -self.clip, self.clip)
                
        # Update return normalization statistics
        if self.normalize_returns:
            self.returns = self.returns * self.gamma + reward
            
            if done:
                self.return_count += 1
                self.return_sum += self.returns
                mean_return = self.return_sum / self.return_count
                norm_reward = reward / (mean_return + self.epsilon)
                if self.clip is not None:
                    norm_reward = np.clip(norm_reward, -self.clip, self.clip)
                    
        return obs, norm_reward, terminated, truncated, info


class TimeLimit(gym.Wrapper):
    """
    Limit episode length to a maximum number of steps.
    
    Attributes:
        max_steps: Maximum number of steps per episode
        current_steps: Current number of steps in episode
    """
    
    def __init__(self, env: gym.Env, max_steps: int):
        """
        Initialize a time limit wrapper.
        
        Args:
            env: Environment to wrap
            max_steps: Maximum number of steps per episode
        """
        super().__init__(env)
        self.max_steps = max_steps
        self.current_steps = 0
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and the step counter."""
        self.current_steps = 0
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step and check if the time limit is reached."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps += 1
        
        # Set truncated to True if time limit is reached
        if self.current_steps >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
            
        return obs, reward, terminated, truncated, info


class FrameStack(gym.Wrapper):
    """
    Stack the most recent frames.
    
    Attributes:
        num_stack: Number of frames to stack
        frames: Deque of frames
    """
    
    def __init__(self, env: gym.Env, num_stack: int = 4):
        """
        Initialize a frame stack wrapper.
        
        Args:
            env: Environment to wrap
            num_stack: Number of frames to stack
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # Update observation space
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype
        )
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and fill the frame stack with initial frame."""
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_stacked_obs(), info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step and update the frame stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info
    
    def _get_stacked_obs(self) -> np.ndarray:
        """Get stacked observations."""
        return np.array(self.frames)


class VecEnv:
    """
    Vectorized environment that runs multiple environments in parallel.
    
    Attributes:
        envs: List of environments
        num_envs: Number of environments
    """
    
    def __init__(self, envs: List[gym.Env]):
        """
        Initialize a vectorized environment.
        
        Args:
            envs: List of environments to run in parallel
        """
        self.envs = envs
        self.num_envs = len(envs)
        
        # Check that all environments have the same observation and action spaces
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        
        for env in envs:
            assert env.observation_space == self.observation_space, "All environments must have the same observation space"
            assert env.action_space == self.action_space, "All environments must have the same action space"
            
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments."""
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = [None] * self.num_envs
            
        results = [env.reset(seed=s, options=options) for env, s in zip(self.envs, seeds)]
        obs, infos = zip(*results)
        
        return np.array(obs), list(infos)
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Take a step in all environments."""
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rewards, terminated, truncated, infos = zip(*results)
        
        return (
            np.array(obs),
            np.array(rewards),
            np.array(terminated),
            np.array(truncated),
            list(infos)
        )
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()