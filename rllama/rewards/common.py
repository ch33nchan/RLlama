from .base import BaseReward
from typing import Any, Dict

class GoalReward(BaseReward):
    """Simple reward for reaching a goal state."""
    @property
    def name(self) -> str:
        return "goal_reward"

    def __init__(self, goal_key: str = "goal_reached", reward_value: float = 1.0):
        self._goal_key = goal_key
        self._reward_value = reward_value

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        return self._reward_value if info.get(self._goal_key, False) else 0.0

class StepPenaltyReward(BaseReward):
    """Applies a constant penalty for each step taken."""
    @property
    def name(self) -> str:
        return "step_penalty"

    def __init__(self, penalty: float = -0.01):
        self._penalty = penalty

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        return self._penalty

class ActionNoveltyReward(BaseReward):
    """Rewards taking less frequent actions (requires tracking)."""
    @property
    def name(self) -> str:
        return "action_novelty"

    def __init__(self, frequency_key: str = "action_frequencies"):
        self._frequency_key = frequency_key

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        action_frequencies = info.get(self._frequency_key, {})
        # Ensure action is hashable or convert to a hashable representation
        try:
            action_key = hash(action)
        except TypeError:
            action_key = hash(str(action)) # Fallback for unhashable actions

        frequency = action_frequencies.get(action_key, 0)
        # Add 1 to frequency in denominator to avoid division by zero and scale reward
        return 1.0 / (1.0 + frequency)