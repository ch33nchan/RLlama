from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseReward(ABC):
    """Abstract base class for individual reward components."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this reward component."""
        pass

    @abstractmethod
    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """
        Calculate the reward value based on the transition.

        Args:
            state: Current environment state.
            action: Action taken.
            next_state: Resulting environment state.
            info: Dictionary containing auxiliary information from the environment step
                  (e.g., goal reached, step count, internal env state).

        Returns:
            The calculated float reward value for this component.
        """
        pass