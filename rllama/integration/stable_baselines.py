

import gym
from typing import Dict, Any, Optional, Callable, Union, Type
from gym import spaces
import numpy as np

try:
    import stable_baselines3 as sb3
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from ..rewards.shaper import RewardShaper
from ..rewards.composer import RewardComposer

class RLlamaWrapper(gym.Wrapper):
    """
    Wrapper for Gym environments to use RLlama reward components.
    Compatible with Stable Baselines3.
    """
    
    def __init__(self, 
                 env: gym.Env, 
                 reward_composer: RewardComposer,
                 reward_shaper: RewardShaper,
                 context_builder: Callable[[Dict[str, Any], np.ndarray, float, bool, Dict[str, Any]], Dict[str, Any]] = None):
        """
        Initialize the wrapper.
        
        Args:
            env: Gym environment to wrap
            reward_composer: RLlama reward composer
            reward_shaper: RLlama reward shaper
            context_builder: Function to build context from env step outputs
        """
        super().__init__(env)
        
        if not SB3_AVAILABLE:
            raise ImportError("Stable Baselines3 is required for this integration. "
                              "Install it with 'pip install stable-baselines3'")
                              
        self.reward_composer = reward_composer
        self.reward_shaper = reward_shaper
        self.context_builder = context_builder or self._default_context_builder
        self.step_counter = 0
        self.episode_counter = 0
        
    def _default_context_builder(self, 
                                obs: np.ndarray, 
                                reward: float, 
                                done: bool, 
                                info: Dict[str, Any]) -> Dict[str, Any]:
        """Default context builder function."""
        context = {
            'observation': obs,
            'env_reward': reward,
            'done': done,
            'step': self.step_counter,
            'episode': self.episode_counter
        }
        
        # Add all info dict contents
        context.update(info)
        
        return context
        
    def step(self, action):
        """
        Step the environment and apply RLlama rewards.
        """
        # Step the original environment
        obs, original_reward, done, info = self.env.step(action)
        
        # Increment step counter
        self.step_counter += 1
        
        # Build context for reward calculation
        context = self.context_builder(obs, original_reward, done, info)
        
        # Add action to context
        context['action'] = action
        
        # Calculate rewards from components
        component_rewards = self.reward_composer.calculate(context)
        
        # Shape rewards
        shaped_reward = self.reward_shaper.shape(component_rewards, self.step_counter)
        
        # If done, increment episode counter
        if done:
            self.episode_counter += 1
            
        return obs, shaped_reward, done, info
        
    def reset(self):
        """Reset the environment and the step counter."""
        self.step_counter = 0
        return self.env.reset()
