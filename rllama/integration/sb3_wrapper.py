import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from rllama.rewards import RewardComposer, RewardShaper, RewardConfigLoader

class SB3RllamaWrapper(gym.Wrapper):
    """
    A Stable Baselines3 wrapper to integrate RLlama reward shaping.
    """
    def __init__(self, env: gym.Env, rllama_config_path: str, 
                 rllama_components: Optional[Dict[str, Any]] = None, # For programmatic component registration
                 pass_full_info_to_rllama: bool = True):
        super().__init__(env)
        self.rllama_config_loader = RewardConfigLoader(config_path=rllama_config_path)
        
        # Load components, composer, shaper from config
        # Assuming RewardConfigLoader has methods to instantiate these
        # Or, you might instantiate them directly here based on loaded config dict
        config_dict = self.rllama_config_loader.load_config()

        # Allow programmatic registration/override of components if needed
        # For simplicity, assuming components are registered globally or handled by RewardConfigLoader
        
        self.composer: RewardComposer = self.rllama_config_loader.create_composer(config_dict)
        self.shaper: RewardShaper = self.rllama_config_loader.create_shaper(config_dict)

        self.pass_full_info_to_rllama = pass_full_info_to_rllama
        self._last_obs = None
        self._last_action = None # Store last action for context

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self._last_obs, info = self.env.reset(**kwargs)
        self._last_action = None # Reset last action on episode start
        self.composer.reset()
        self.shaper.reset()
        return self._last_obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        next_obs, base_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        rllama_context = {
            "state": self._last_obs,
            "action": action,
            "next_state": next_obs,
            "base_reward": base_reward,
            "done": done,
            "info": info if self.pass_full_info_to_rllama else {},
            "previous_action": self._last_action, # Example of adding more context
            # Add other relevant context items your components might need
        }
        
        # Get raw and normalized rewards from composer
        _, raw_comp_rewards, norm_comp_rewards = self.composer.calculate_reward(**rllama_context)
        
        # Get final shaped reward from shaper
        shaped_reward = self.shaper.shape_reward(
            component_rewards=norm_comp_rewards, # Shaper works on normalized rewards
            base_reward=base_reward # Shaper can optionally include the base_reward
        )

        # Store raw, normalized, and weighted rewards in info if desired
        info["rllama_raw_rewards"] = raw_comp_rewards
        info["rllama_normalized_rewards"] = norm_comp_rewards
        info["rllama_weighted_rewards"] = self.shaper.get_last_weighted_rewards()
        info["rllama_total_shaped_reward"] = shaped_reward
        info["rllama_base_reward"] = base_reward
        
        self._last_obs = next_obs
        self._last_action = action # Store current action as previous for next step

        return next_obs, float(shaped_reward), terminated, truncated, info

# Example Usage (conceptual):
# from stable_baselines3 import PPO
# from rllama.utils.config_loader import register_component # if needed
# # from rllama.rewards.robotics_components import TargetReachedReward # etc.
# # register_component("TargetReachedReward", TargetReachedReward)

# env = gym.make("YourEnv-v0")
# wrapped_env = SB3RllamaWrapper(env, "path/to/your/rllama_config.yaml")
# model = PPO("MlpPolicy", wrapped_env, verbose=1)
# model.learn(total_timesteps=10000)