# rllama/rewards/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseReward(ABC):
    """
    Abstract base class for all reward components.

    Each component must implement the `calculate` method. The `__init__`
    method can be overridden in subclasses to accept specific parameters
    from the YAML configuration.
    """

    def __init__(self, **kwargs):
        """
        The constructor can accept any parameters defined under the `params`
        key for this component in your YAML config. `kwargs` will be a
        dictionary containing these parameters.
        """
        pass

    @abstractmethod
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculates the reward for a given context.

        This method must be implemented by all subclasses.

        Args:
            context (Dict[str, Any]): A dictionary containing all necessary
                                     information for the reward calculation,
                                     e.g., {'prompt': str, 'response': str,
                                     'info': dict}.

        Returns:
            float: The calculated reward value for this component.
        """
        pass