from typing import List, Dict, Any, Optional # Add Optional
from .base import BaseReward
import logging

logger = logging.getLogger(__name__)

class RewardComposer:
    """
    Combines rewards from multiple components and applies optional normalization and shaping.
    """
    # Update __init__ signature
    def __init__(self, reward_components: List[BaseReward], normalization_strategy: Optional[str] = None):
        """
        Initializes the RewardComposer.

        Args:
            reward_components: A list of BaseReward component instances.
            normalization_strategy: The strategy for normalizing raw rewards (e.g., 'min_max', 'z_score', None).
        """
        if not reward_components:
            logger.warning("RewardComposer initialized with no reward components.")
        # Store the components and strategy
        self.components = reward_components
        self.normalization_strategy = normalization_strategy
        self._component_names = [c.name for c in reward_components] # Store names for convenience
        logger.info(f"RewardComposer initialized with components: {self._component_names}, Normalization: {normalization_strategy}")


    def compute_rewards(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes the raw reward value from each component.

        Args:
            state: The state before the action.
            action: The action taken.
            next_state: The state after the action.
            info: A dictionary containing auxiliary information.

        Returns:
            A dictionary mapping component names to their calculated raw reward values.
        """
        raw_rewards = {}
        for component in self.components:
            try:
                reward = component(state, action, next_state, info)
                raw_rewards[component.name] = reward
            except Exception as e:
                logger.error(f"Error computing reward for component '{component.name}': {e}", exc_info=True)
                raw_rewards[component.name] = 0.0 # Assign 0 reward on error? Or raise?
        return raw_rewards

    def _normalize_rewards(self, raw_rewards: Dict[str, float]) -> Dict[str, float]:
        """Applies the chosen normalization strategy."""
        if self.normalization_strategy is None:
            return raw_rewards
        elif self.normalization_strategy == 'min_max':
            # TODO: Implement min-max normalization (requires tracking min/max values)
            logger.warning("Min-max normalization not yet implemented.")
            return raw_rewards
        elif self.normalization_strategy == 'z_score':
            # TODO: Implement z-score normalization (requires tracking mean/stddev)
            logger.warning("Z-score normalization not yet implemented.")
            return raw_rewards
        else:
            logger.warning(f"Unknown normalization strategy: {self.normalization_strategy}. Skipping normalization.")
            return raw_rewards

    def combine_rewards(self, raw_rewards: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Combines the raw (potentially normalized) rewards using provided weights.

        Args:
            raw_rewards: A dictionary mapping component names to raw reward values.
            weights: A dictionary mapping component names to their current weights.

        Returns:
            The final combined scalar reward.
        """
        normalized_rewards = self._normalize_rewards(raw_rewards)
        final_reward = 0.0
        for name, reward in normalized_rewards.items():
            weight = weights.get(name)
            if weight is None:
                logger.warning(f"No weight found for reward component '{name}'. Skipping.")
                continue
            if weight < 0:
                 logger.warning(f"Negative weight ({weight}) applied to component '{name}'.")

            final_reward += reward * weight

        # Log the components, raw values, weights, and final reward for debugging
        # logger.debug(f"Combining rewards: Raw={raw_rewards}, Norm={normalized_rewards}, Weights={weights}, Final={final_reward}")

        return final_reward

    def get_reward(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        Computes and combines rewards in one step.

        Args:
            state: The state before the action.
            action: The action taken.
            next_state: The state after the action.
            info: A dictionary containing auxiliary information.
            weights: A dictionary mapping component names to their current weights.

        Returns:
            The final combined scalar reward.
        """
        raw_rewards = self.compute_rewards(state, action, next_state, info)
        final_reward = self.combine_rewards(raw_rewards, weights)
        return final_reward