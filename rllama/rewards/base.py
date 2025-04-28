from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseReward(ABC):
    """Abstract base class for all reward components."""

    def __init__(self, name: str):
        """
        Initializes the base reward component.

        Args:
            name: The unique name identifier for this reward component.
        """
        if not name:
            raise ValueError("Reward component name cannot be empty.")
        self._name = name # Store the name

    @property
    @abstractmethod # Make name an abstract property if subclasses must implement it (though __init__ handles it now)
    def name(self) -> str:
        """Returns the name of the reward component."""
        # If subclasses MUST implement the property getter, keep abstractmethod
        # If __init__ is sufficient, this property could be concrete here:
        # return self._name
        # Let's assume subclasses should still define it for clarity or potential overrides
        pass # Keep abstract for now, subclasses provide concrete @property

    @abstractmethod
    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """
        Calculates the reward value based on the given transition.

        Args:
            state: The state before the action.
            action: The action taken.
            next_state: The state after the action.
            info: A dictionary containing auxiliary information (e.g., metrics).

        Returns:
            The calculated reward value (float).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"