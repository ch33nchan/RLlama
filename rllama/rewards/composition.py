from typing import List, Dict, Any, Optional, Tuple # Add Tuple
from .base import BaseReward
import logging
import math # Add math for variance calculation

logger = logging.getLogger(__name__)

# Helper class for running statistics (for z-score)
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0 # Sum of squares of differences from the current mean

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean # New delta after mean update
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

# Helper class for running min/max
class RunningMinMax:
    def __init__(self):
        self.min = float('inf')
        self.max = float('-inf')
        self.n = 0

    def update(self, x: float):
        self.n += 1
        self.min = min(self.min, x)
        self.max = max(self.max, x)

    @property
    def range(self) -> float:
        return self.max - self.min if self.n > 0 and self.max > self.min else 1.0 # Avoid division by zero


class RewardComposer:
    """
    Combines rewards from multiple components and applies optional normalization and shaping.
    """
    # Update __init__ signature
    def __init__(self, reward_components: List[BaseReward], normalization_strategy: Optional[str] = None, norm_epsilon: float = 1e-8, norm_warmup_steps: int = 10):
        """
        Initializes the RewardComposer.

        Args:
            reward_components: A list of BaseReward component instances.
            normalization_strategy: The strategy for normalizing raw rewards ('min_max', 'z_score', None).
            norm_epsilon: Small value added to stddev/range denominator for numerical stability.
            norm_warmup_steps: Number of steps before applying normalization to allow stats to stabilize.
        """
        if not reward_components:
            logger.warning("RewardComposer initialized with no reward components.")
        # Store the components and strategy
        self.components = reward_components
        self.normalization_strategy = normalization_strategy
        self.norm_epsilon = norm_epsilon
        self.norm_warmup_steps = norm_warmup_steps
        self._component_names = [c.name for c in reward_components] # Store names for convenience

        # Initialize statistics storage based on strategy
        self._reward_stats: Dict[str, Any] = {}
        if self.normalization_strategy == 'z_score':
            self._reward_stats = {name: RunningStats() for name in self._component_names}
            logger.info("Initialized running z-score statistics.")
        elif self.normalization_strategy == 'min_max':
            self._reward_stats = {name: RunningMinMax() for name in self._component_names}
            logger.info("Initialized running min-max statistics.")
        elif self.normalization_strategy is not None:
             logger.warning(f"Unknown normalization strategy: {self.normalization_strategy}. Will not normalize.")
             self.normalization_strategy = None # Disable unknown strategy

        logger.info(f"RewardComposer initialized with components: {self._component_names}, Normalization: {self.normalization_strategy}, Warmup: {self.norm_warmup_steps}")


    def compute_rewards(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Computes the raw reward value from each component and updates normalization stats.

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
            name = component.name
            try:
                reward = component(state, action, next_state, info)
                raw_rewards[name] = reward
                # Update stats if normalization is active
                if self.normalization_strategy and name in self._reward_stats:
                    self._reward_stats[name].update(reward)
            except Exception as e:
                logger.error(f"Error computing reward for component '{name}': {e}", exc_info=True)
                raw_rewards[name] = 0.0 # Assign 0 reward on error? Or raise?
        return raw_rewards

    def _normalize_rewards(self, raw_rewards: Dict[str, float]) -> Dict[str, float]:
        """Applies the chosen normalization strategy using running statistics."""
        if self.normalization_strategy is None:
            return raw_rewards

        normalized_rewards = {}
        for name, reward in raw_rewards.items():
            stats = self._reward_stats.get(name)
            if stats is None:
                logger.warning(f"No stats found for component '{name}' during normalization. Skipping.")
                normalized_rewards[name] = reward
                continue

            # Skip normalization during warmup period
            if stats.n < self.norm_warmup_steps:
                 normalized_rewards[name] = reward # Return raw reward during warmup
                 continue

            if self.normalization_strategy == 'min_max':
                if isinstance(stats, RunningMinMax):
                    # Normalize to [0, 1] range
                    norm_range = stats.range
                    normalized_rewards[name] = (reward - stats.min) / (norm_range + self.norm_epsilon)
                else:
                     logger.error(f"Mismatched stats type for min_max: {type(stats)}. Skipping normalization for {name}.")
                     normalized_rewards[name] = reward

            elif self.normalization_strategy == 'z_score':
                if isinstance(stats, RunningStats):
                    stddev = stats.stddev
                    normalized_rewards[name] = (reward - stats.mean) / (stddev + self.norm_epsilon)
                else:
                    logger.error(f"Mismatched stats type for z_score: {type(stats)}. Skipping normalization for {name}.")
                    normalized_rewards[name] = reward
            else:
                # Should not happen due to check in __init__, but safeguard
                normalized_rewards[name] = reward

        return normalized_rewards

    def combine_rewards(self, raw_rewards: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Combines the raw (potentially normalized) rewards using provided weights.
        Args:
            raw_rewards: A dictionary mapping component names to raw reward values.
            weights: A dictionary mapping component names to their current weights. (<<< From RewardShaper)
        Returns:
            The final combined scalar reward.
        """
        processed_rewards = self._normalize_rewards(raw_rewards) # Optional normalization
        final_reward = 0.0
        for name, reward in processed_rewards.items():
            weight = weights.get(name) # <<< Gets the current weight for this component
            if weight is None:
                logger.warning(f"No weight found for reward component '{name}'. Skipping.")
                continue
            if weight < 0:
                 logger.warning(f"Negative weight ({weight}) applied to component '{name}'.")

            weighted_reward = reward * weight # <<< Applies the weight
            final_reward += weighted_reward
        return final_reward

    def get_reward(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        Computes, normalizes (if configured), and combines rewards in one step.

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
        # Normalization happens within combine_rewards now
        final_reward = self.combine_rewards(raw_rewards, weights) # <<< Uses weights here
        return final_reward # <<< Corrected indentation (removed leading space)

    # Add a method to get current stats for inspection/debugging if needed
    def get_normalization_stats(self) -> Dict[str, Dict[str, Any]]:
        """Returns the current normalization statistics for inspection."""
        stats_summary = {}
        if self.normalization_strategy == 'z_score':
            for name, stats in self._reward_stats.items():
                if isinstance(stats, RunningStats):
                    stats_summary[name] = {'n': stats.n, 'mean': stats.mean, 'stddev': stats.stddev}
        elif self.normalization_strategy == 'min_max':
             for name, stats in self._reward_stats.items():
                 if isinstance(stats, RunningMinMax):
                    stats_summary[name] = {'n': stats.n, 'min': stats.min, 'max': stats.max}
        return stats_summary

    # Remove the entire some_method block
    # def some_method(self, data):
    #    result = self._process_data(data)
    #    logger.debug(f"Processed data result: {result}") # <<< Line 193: Correct indentation