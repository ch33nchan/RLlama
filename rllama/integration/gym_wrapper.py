# rllama/integration/gym_wrapper.py

import gym
from typing import Dict, Any, Optional, Callable, Union, List
import numpy as np

from ..engine import RewardEngine

class RLlamaGymWrapper(gym.Wrapper):
    """
    A wrapper for OpenAI Gym environments that uses RLlama for reward shaping.
    """
    
    def __init__(self, 
                 env: gym.Env, 
                 reward_engine: RewardEngine,
                 context_mapper: Optional[Callable] = None,
                 combine_rewards: bool = True):
        """
        Initialize the wrapper.
        
        Args:
            env: The gym environment to wrap
            reward_engine: The RLlama reward engine
            context_mapper: Function to map env observations to RLlama context
            combine_rewards: Whether to add RLlama reward to env reward
        """
        super().__init__(env)
        self.reward_engine = reward_engine
        self.context_mapper = context_mapper or self._default_mapper
        self.combine_rewards = combine_rewards
        self.step_counter = 0
        self.episode_counter = 0
        
    def _default_mapper(self, 
                       observation: np.ndarray, 
                       action: np.ndarray, 
                       reward: float, 
                       done: bool, 
                       info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default mapping from gym step outputs to RLlama context.
        """
        context = {
            'observation': observation,
            'action': action,
            'env_reward': reward,
            'done': done,
            'info': info,
            'step': self.step_counter,
            'episode': self.episode_counter
        }
        return context
        
    def step(self, action):
        """
        Step the environment and compute shaped rewards.
        """
        # Step the environment
        obs, reward, done, info = self.env.step(action)
        
        # Increment step counter
        self.step_counter += 1
        
        # Map to RLlama context
        context = self.context_mapper(obs, action, reward, done, info)
        
        # Compute RLlama reward
        rllama_reward = self.reward_engine.compute(context)
        
        # Combine rewards if specified
        if self.combine_rewards:
            final_reward = reward + rllama_reward
        else:
            final_reward = rllama_reward
            
        # Update info with reward components
        info['rllama_reward'] = rllama_reward
        info['original_reward'] = reward
        
        # Reset step counter if done
        if done:
            self.episode_counter += 1
            
        return obs, final_reward, done, info
        
    def reset(self, **kwargs):
        """Reset the environment and internal counters."""
        self.step_counter = 0
        return self.env.reset(**kwargs)
