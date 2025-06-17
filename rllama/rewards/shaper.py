# rllama/rewards/shaper.py

from typing import Dict

class RewardShaper:
    """
    Applies weights from a configuration to a dictionary of raw
    component rewards to produce a final, single scalar reward.
    
    This class can be extended to handle advanced reward shaping,
    such as scheduling, where weights change over time.
    """
    def __init__(self, shaping_config: Dict):
        """
        Args:
            shaping_config (Dict): The 'shaping_config' block from your
                                   YAML file.
        """
        self.config = shaping_config

    def shape(self, component_rewards: Dict[str, float], step: int = 0) -> float:
        """
        Shapes the final reward by applying weights and summing the components.

        Args:
            component_rewards (Dict[str, float]): A dict of raw reward values
                                                 from the RewardComposer.
            step (int): The current training step, for optional scheduling.

        Returns:
            The final shaped reward as a single float.
        """
        final_reward = 0.0
        for name, reward_val in component_rewards.items():
            # Get weight from config, default to 1.0 if the component is
            # not listed in the shaping_config block.
            weight = self.config.get(name, {}).get('weight', 1.0)
            final_reward += weight * reward_val

        return final_rewardw