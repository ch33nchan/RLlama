// ... existing code ...
from .config import RewardConfig
import logging

logger = logging.getLogger(__name__)

class RewardShaper:
    """
    Applies weighting, scheduling, and final composition to rewards.
    """
    def __init__(self,
                 reward_configs: Dict[str, RewardConfig],
                 composition_strategy: Literal['additive', 'multiplicative'] = 'additive',
                 default_weight: float = 1.0):
        """
        Initializes the RewardShaper.

        Args:
            reward_configs: A dictionary mapping component names to their RewardConfig objects.
            composition_strategy: How to combine the weighted component rewards.
                                  'additive': Sum of weighted component rewards.
                                  'multiplicative': Product of (1 + weighted_component_rewards) for positive,
                                                  (1 - abs(weighted_component_rewards)) for negative.
                                                  This helps ensure a zero reward doesn't nullify everything
                                                  if it's meant to be neutral.
            default_weight: Default weight for components not in reward_configs.
        """
        self.reward_configs = reward_configs
        self.composition_strategy = composition_strategy
        self.default_weight = default_weight
        self.current_step = 0
        self._last_weighted_rewards: Dict[str, float] = {}
        logger.info(f"RewardShaper initialized with composition strategy: {self.composition_strategy}")

    def _get_weight(self, component_name: str) -> float:
// ... existing code ...
    def shape_reward(self,
                     component_rewards: Dict[str, float],
                     base_reward: Optional[float] = None,
                     normalization_stats: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
        """
        Shapes the reward based on component rewards, their configurations, and an optional base reward.

        Args:
            component_rewards: A dictionary of {component_name: reward_value}
                               These are typically the *normalized* rewards from RewardComposer.
            base_reward: An optional base reward from the environment to be included.
            normalization_stats: Optional dictionary of normalization stats for logging/debugging.

        Returns:
            The final shaped reward.
        """
        self._last_weighted_rewards.clear()
        weighted_rewards_values = []

        for name, reward_val in component_rewards.items():
            weight = self._get_weight(name)
            weighted_reward = weight * reward_val
            self._last_weighted_rewards[name] = weighted_reward
            weighted_rewards_values.append(weighted_reward)

        if not weighted_rewards_values and base_reward is None:
            final_reward = 0.0
        elif self.composition_strategy == 'additive':
            final_reward = sum(weighted_rewards_values)
            if base_reward is not None:
                final_reward += base_reward
        elif self.composition_strategy == 'multiplicative':
            # Transform rewards: R' = 1 + R for R >= 0, R' = 1 / (1 + abs(R)) for R < 0
            # This maps positive rewards > 0 to > 1, negative rewards < 0 to (0, 1), and 0 to 1.
            # Then take product. If product is P, final reward is P-1.
            # This is one way to handle multiplicative composition where components can be positive or negative.
            # A simpler (1+R) approach for positive rewards, (1-|R|) for negative rewards.
            # Let's use: if R > 0, use 1+R. If R < 0, use max(epsilon, 1-|R|). If R = 0, use 1.
            # This keeps positive contributions multiplicative and negative ones reductive.
            
            # Simpler approach: if reward is r, use (1+r) for product, then subtract 1 from total product.
            # This means a 0 weighted reward becomes a factor of 1 (neutral).
            # A positive reward r becomes (1+r) > 1.
            # A negative reward r becomes (1+r) < 1.
            # Example: r1=0.5, r2=0.2 -> (1.5)*(1.2) = 1.8. Final = 0.8 (sum would be 0.7)
            # Example: r1=0.5, r2=-0.2 -> (1.5)*(0.8) = 1.2. Final = 0.2 (sum would be 0.3)
            # Example: r1=-0.5, r2=-0.2 -> (0.5)*(0.8) = 0.4. Final = -0.6 (sum would be -0.7)
            
            if not weighted_rewards_values:
                final_reward = 0.0
            else:
                product_reward = 1.0
                for wr_val in weighted_rewards_values:
                    product_reward *= (1 + wr_val) # Assumes wr_val is not too negative (e.g. not < -1)
                                                # We might need to clip wr_val or use a safer transform.
                                                # For instance, ensure (1+wr_val) is positive.
                                                # Let's assume weights and normalized rewards keep this reasonable.
                                                # A more robust approach:
                                                # if wr_val >= 0: factor = 1 + wr_val
                                                # else: factor = 1.0 / (1 + abs(wr_val)) # makes it reductive
                    product_reward *= max(1e-6, 1 + wr_val) # Ensure factor is positive

                final_reward = product_reward - 1.0 # Subtract the initial 1s

            if base_reward is not None:
                # How to combine base_reward with multiplicative? Additively is simplest.
                final_reward += base_reward
        else:
            logger.warning(f"Unknown composition strategy: {self.composition_strategy}. Defaulting to additive.")
            final_reward = sum(weighted_rewards_values)
            if base_reward is not None:
                final_reward += base_reward
        
        self.current_step += 1
        return final_reward

// ... existing code ...