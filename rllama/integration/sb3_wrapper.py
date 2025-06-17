# rllama/integration/sb3_wrapper.py

import gymnasium as gym
from typing import Any, SupportsFloat

from rllama.engine import RewardEngine

class SB3RllamaRewardWrapper(gym.Wrapper):
    """
    A Gymnasium wrapper to integrate the RLlama RewardEngine with Stable Baselines 3.

    This wrapper intercepts the reward from the base environment at each step,
    computes a supplemental reward using the RLlama engine, and adds it to
    the original reward.
    """
    def __init__(self, env: gym.Env, rllama_config_path: str):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            rllama_config_path (str): The file path to the RLlama YAML config.
        """
        super().__init__(env)
        print("Initializing RLlama for Stable Baselines 3...")
        # This uses the RewardEngine we defined previously
        self.reward_engine = RewardEngine(config_path=rllama_config_path)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Overrides the environment's step method to inject the RLlama reward.
        """
        # 1. Get the original transition from the base environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # 2. Construct the context for RLlama
        # The 'info' dictionary from the environment is a great place to put
        # any information that reward components might need.
        context = {
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

        # 3. Compute the supplemental reward from RLlama and log the trace
        rllama_reward = self.reward_engine.compute_and_log(context)

        # 4. Combine the rewards
        # We add the RLlama reward to the environment's base reward.
        # This allows RLlama to act as a "bonus" or shaping signal.
        combined_reward = reward + rllama_reward
        
        # Add rllama-specific info for debugging or analysis
        info['rllama_reward'] = rllama_reward
        info['original_reward'] = reward

        return observation, combined_reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Resets the environment and the RLlama engine's step counter."""
        self.reward_engine.current_step = 0
        return self.env.reset(**kwargs)