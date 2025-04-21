import torch
from typing import List, Dict, Any
from .base import BaseReward # Import the base class

class RewardComposer:
    def __init__(self, reward_components: List[BaseReward]):
        """
        Initializes the composer with a list of BaseReward instances.
        """
        self.reward_components = {comp.name: comp for comp in reward_components}
        if len(self.reward_components) != len(reward_components):
             # This check helps catch duplicate reward names early
             raise ValueError("Duplicate reward names detected in reward_components.")

    def compute_rewards(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes all individual raw reward values using the registered components.
        """
        raw_rewards = {}
        for name, component in self.reward_components.items():
            raw_rewards[name] = component(state, action, next_state, info) # Call the component instance
        return raw_rewards

    def combine_rewards(self, raw_rewards: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Combines raw rewards using provided weights.
        """
        total_reward = 0.0
        for name, value in raw_rewards.items():
            weight = weights.get(name)
            if weight is None:
                print(f"Warning: No weight found for reward component '{name}'. Skipping.")
                continue
            total_reward += weight * value
        return total_reward