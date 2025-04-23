import torch
from typing import List, Dict, Any
from .base import BaseReward # Import the base class
from collections import deque
import numpy as np

class RewardComposer:
    def __init__(self, components: List[BaseReward], normalize: bool = False, norm_window: int = 1000):
         """
         Initializes the composer with a list of BaseReward instances
         and optionally sets up normalization.
         """
         # Use self.reward_components consistently, or rename component_map
         # Let's stick to self.reward_components for the map
         self.reward_components = {comp.name: comp for comp in components}
         if len(self.reward_components) != len(components):
             # This check helps catch duplicate reward names early
             raise ValueError("Duplicate reward names detected in reward_components.")

         # Setup normalization if requested
         self.normalize = normalize
         if self.normalize:
             self._reward_stats = {name: {'mean': 0.0, 'std': 1.0, 'history': deque(maxlen=norm_window)}
                                   for name in self.reward_components} # Use self.reward_components here

    def _update_stats(self, name: str, value: float):
        """Update running mean and std."""
        history = self._reward_stats[name]['history']
        history.append(value)
        if len(history) > 1:
             self._reward_stats[name]['mean'] = np.mean(history)
             self._reward_stats[name]['std'] = np.std(history) + 1e-8 # Add epsilon for stability

    def compute_rewards(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> Dict[str, float]:
        """Computes raw rewards from all components."""
        raw_rewards = {}
        # Use self.reward_components here as well for consistency
        for name, component in self.reward_components.items():
            try:
                value = component(state, action, next_state, info)
                raw_rewards[name] = value
                if self.normalize:
                    # Update stats *after* getting the value, before potential normalization in combine_rewards
                    self._update_stats(name, value)
            except Exception as e:
                # Handle potential errors in individual components gracefully
                print(f"Warning: Error computing reward for '{name}': {e}")
                raw_rewards[name] = 0.0
        return raw_rewards

    def combine_rewards(self, raw_rewards: Dict[str, float], weights: Dict[str, float]) -> float:
        """Combines raw rewards using provided weights, applying normalization if enabled."""
        final_reward = 0.0
        for name, raw_value in raw_rewards.items():
            if name not in weights:
                print(f"Warning: No weight found for reward component '{name}'. Skipping.")
                continue

            value_to_combine = raw_value
            if self.normalize:
                # Ensure stats exist before accessing, though __init__ should guarantee it if self.normalize is True
                if name in self._reward_stats:
                    stats = self._reward_stats[name]
                    # Normalize: (value - mean) / std
                    value_to_combine = (raw_value - stats['mean']) / stats['std']
                else:
                     print(f"Warning: Normalization enabled but no stats found for '{name}'. Using raw value.")


            final_reward += value_to_combine * weights[name]
        return final_reward

    def get_component_names(self):
        # Use self.reward_components here
        return list(self.reward_components.keys())