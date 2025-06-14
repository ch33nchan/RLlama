# rllama/rewards/components/common.py

from typing import Dict, Any
from rllama.rewards.base import BaseReward

class LengthReward(BaseReward):
    """
    Calculates a reward based on the length of the response.
    It applies a quadratic penalty for the deviation from a target length.
    """
    def __init__(self, target_length: int = 100, strength: float = 0.001):
        """
        Args:
            target_length (int): The desired length of the response.
            strength (float): The strength of the penalty.
        """
        super().__init__()
        self.target_length = target_length
        self.strength = strength

    def calculate(self, context: Dict[str, Any]) -> float:
        response = context.get("response", "")
        if not isinstance(response, str):
            return 0.0
        
        # Quadratic penalty for deviation from target length
        return -self.strength * ((len(response) - self.target_length) ** 2)

class ConstantReward(BaseReward):
    """
    Provides a constant reward value at each step.
    Useful for encouraging agent survival or penalizing actions.
    """
    def __init__(self, value: float = 0.01):
        """
        Args:
            value (float): The constant reward value to return.
        """
        super().__init__()
        self.value = value
        
    def calculate(self, context: Dict[str, Any]) -> float:
        return self.value