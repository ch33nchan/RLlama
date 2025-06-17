# rllama/integration/sb3_wrapper.py

import gymnasium as gym
from typing import Any, SupportsFloat
from ..engine import RewardEngine # Use relative import from parent directory

class SB3RllamaRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, rllama_config_path: str):
        super().__init__(env)
        self.reward_engine = RewardEngine(config_path=rllama_config_path)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        context = {
            "action": action,
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }
        rllama_reward = self.reward_engine.compute_and_log(context)
        combined_reward = reward + rllama_reward
        info['rllama_reward'] = rllama_reward
        info['original_reward'] = reward
        return observation, combined_reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        self.reward_engine.current_step = 0
        return self.env.reset(**kwargs)