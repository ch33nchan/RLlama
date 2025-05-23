from typing import Dict, Any, Optional

class BaseReward:
    """
    Base class for all reward components.
    Each reward component must inherit from this class.
    """
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initializes the reward component.

        Args:
            name (str): The unique name of the reward component.
                        This is typically the key used in configuration files.
            weight (float): The weight assigned to this reward component,
                            used by composers to scale its contribution.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Reward component name must be a non-empty string.")
        if not isinstance(weight, (int, float)):
            raise ValueError("Reward component weight must be a number.")
            
        self.name = name
        self.weight = weight

    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculates the reward value based on the given context.
        Subclasses MUST implement this method.

        Args:
            context (Dict[str, Any]): A dictionary containing all necessary
                                      information to calculate the reward.
        Returns:
            float: The calculated reward value.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'calculate' method.")

    def reset(self):
        """
        Resets any internal state of the reward component.
        Optional: Subclasses can override this.
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"

    def __repr__(self) -> str:
        return self.__str__()