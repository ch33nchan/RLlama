import gym
import numpy as np
from typing import Dict, Any, Optional, List
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import yaml

from ..core.composer import RewardComposer
from ..core.shaper import RewardShaper

class RLlamaRewardWrapper(gym.Wrapper):
    """
    Stable Baselines3 wrapper for RLlama reward processing
    """
    
    def __init__(self, env: gym.Env, config_path: str):
        super().__init__(env)
        
        # Load RLlama configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize RLlama components
        self.composer = RewardComposer(config.get('composer', {}))
        self.shaper = RewardShaper(config.get('shaper', {}))
        
        # Track statistics
        self.episode_rewards = []
        self.step_count = 0
    
    def step(self, action):
        """Step function with RLlama reward processing"""
        obs, reward, done, info = self.env.step(action)
        
        # Extract text information for reward calculation
        prompt = info.get('prompt', '')
        response = info.get('response', '')
        
        if prompt and response:
            # Calculate RLlama reward
            rllama_reward = self.composer.compose(prompt, response)
            shaped_reward = self.shaper.shape([rllama_reward])[0]
            
            # Replace or augment original reward
            reward = shaped_reward
            
            # Add RLlama info
            info['rllama_reward'] = rllama_reward
            info['shaped_reward'] = shaped_reward
        
        self.step_count += 1
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """Reset environment and RLlama components"""
        obs = self.env.reset(**kwargs)
        
        # Reset step count for new episode
        self.step_count = 0
        
        return obs

class RLlamaCallback(BaseCallback):
    """
    Callback for tracking RLlama metrics during SB3 training
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rllama_rewards = []
        self.shaped_rewards = []
    
    def _on_step(self) -> bool:
        """Called after each step"""
        # Extract RLlama information from info
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'rllama_reward' in info:
                self.rllama_rewards.append(info['rllama_reward'])
            if 'shaped_reward' in info:
                self.shaped_rewards.append(info['shaped_reward'])
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        if self.rllama_rewards:
            mean_rllama = np.mean(self.rllama_rewards)
            mean_shaped = np.mean(self.shaped_rewards)
            
            self.logger.record("rllama/mean_reward", mean_rllama)
            self.logger.record("rllama/mean_shaped_reward", mean_shaped)