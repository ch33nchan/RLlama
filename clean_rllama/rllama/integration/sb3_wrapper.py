# rllama/integration/sb3_wrapper.py

import gym
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from gym import spaces
from ..engine import RewardEngine

class SB3RllamaRewardWrapper(gym.Wrapper):
    """
    Wrapper for integrating RLlama with Stable Baselines 3.
    Processes rewards for SB3-based reinforcement learning.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 reward_config_path: str,
                 observation_key: str = None,
                 action_key: str = None,
                 reward_key: str = None,
                 normalize: bool = True):
        """
        Initialize the SB3 reward wrapper.
        
        Args:
            env: The environment to wrap
            reward_config_path: Path to the RLlama reward configuration
            observation_key: Key for observation in the context if env produces dict observations
            action_key: Key for action in the context if env expects dict actions
            reward_key: Key for the default reward in the context
            normalize: Whether to normalize rewards
        """
        super().__init__(env)
        self.reward_engine = RewardEngine(reward_config_path)
        self.observation_key = observation_key
        self.action_key = action_key
        self.reward_key = reward_key or "env_reward"
        self.normalize = normalize
        
        # Initialize normalization statistics
        self.reward_stats = {
            "count": 0,
            "mean": 0.0,
            "var": 0.0,
            "min": float("inf"),
            "max": float("-inf")
        }
        
        # Store last action, observation, and info for reward calculation
        self.last_obs = None
        self.last_action = None
        self.last_info = {}
        self.step_count = 0
        self.episode_count = 0
    
    def reset(self, **kwargs):
        """Reset the environment and internal state"""
        obs = self.env.reset(**kwargs)
        self.last_obs = obs
        self.last_action = None
        self.last_info = {}
        self.step_count = 0
        self.episode_count += 1
        return obs
    
    def step(self, action):
        """
        Step the environment and apply custom reward logic.
        
        Args:
            action: The action to take
            
        Returns:
            (observation, reward, done, info) tuple with modified reward
        """
        # Save action for reward calculation
        self.last_action = action
        
        # Step the original environment
        obs, reward, done, info = self.env.step(action)
        
        # Create context for RLlama reward calculation
        context = self._create_reward_context(self.last_obs, action, obs, reward, done, info)
        
        # Calculate RLlama reward
        rllama_reward = self.reward_engine.compute_and_log(context)
        
        # Store original reward in info
        info['original_reward'] = reward
        info['rllama_reward'] = rllama_reward
        
        # Use RLlama reward instead
        reward = rllama_reward
        
        # Normalize if needed
        if self.normalize:
            reward = self._normalize_reward(reward)
        
        # Update for next step
        self.last_obs = obs
        self.last_info = info
        self.step_count += 1
        
        return obs, reward, done, info
    
    def _create_reward_context(self, 
                               prev_obs: Any, 
                               action: Any, 
                               obs: Any, 
                               reward: float, 
                               done: bool,
                               info: Dict) -> Dict[str, Any]:
        """
        Create a context dictionary for reward calculation.
        
        Args:
            prev_obs: Previous observation
            action: Action taken
            obs: New observation
            reward: Original environment reward
            done: Whether episode is done
            info: Additional information from the environment
            
        Returns:
            Context dictionary for reward calculation
        """
        context = {
            self.reward_key: reward,
            'done': done,
            'info': info,
            'step': self.step_count,
            'episode': self.episode_count
        }
        
        # Add observation
        if self.observation_key:
            context[self.observation_key] = obs
        else:
            context['observation'] = obs
            context['previous_observation'] = prev_obs
        
        # Add action
        if self.action_key:
            context[self.action_key] = action
        else:
            context['action'] = action
        
        # Add additional info if available
        context.update({k: v for k, v in info.items() if k not in context})
        
        return context
    
    def _normalize_reward(self, reward: float) -> float:
        """
        Normalize rewards using running statistics.
        
        Args:
            reward: The raw reward value
            
        Returns:
            Normalized reward value
        """
        # Update statistics
        self.reward_stats["count"] += 1
        old_mean = self.reward_stats["mean"]
        
        # Update mean and variance (Welford's algorithm)
        delta = reward - old_mean
        self.reward_stats["mean"] += delta / self.reward_stats["count"]
        
        if self.reward_stats["count"] > 1:
            delta2 = reward - self.reward_stats["mean"]
            self.reward_stats["var"] += delta * delta2
        
        # Update min/max
        self.reward_stats["min"] = min(self.reward_stats["min"], reward)
        self.reward_stats["max"] = max(self.reward_stats["max"], reward)
        
        # Perform normalization
        if self.reward_stats["count"] > 1 and np.sqrt(self.reward_stats["var"] / (self.reward_stats["count"] - 1)) > 1e-8:
            std = np.sqrt(self.reward_stats["var"] / (self.reward_stats["count"] - 1))
            normalized_reward = (reward - self.reward_stats["mean"]) / std
            return normalized_reward
        
        # If insufficient data, just return the original reward
        return reward