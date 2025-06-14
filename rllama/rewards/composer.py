# rllama/rewards/composer.py

from typing import List, Dict, Any
from rllama.rewards.base import BaseReward

class RewardComposer:
    """
    Manages a list of reward components and calculates their raw,
    unweighted reward values for a given context.
    """
    def __init__(self, reward_components: List[BaseReward]):
        """
        Args:
            reward_components (List[BaseReward]): A list of instantiated
                                                  reward component objects.
        """
        self.reward_components = reward_components

    def calculate(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates the raw reward for each component.

        Args:
            context (Dict[str, Any]): The context dictionary to pass to each
                                     reward component's calculate() method.

        Returns:
            A dictionary mapping each component's class name to its
            calculated raw reward value.
            e.g., {'LengthReward': -0.5, 'ConstantReward': 0.01}
        """
        component_rewards = {}
        for component in self.reward_components:
            component_name = component.__class__.__name__
            component_rewards[component_name] = component.calculate(context)

        return component_rewards