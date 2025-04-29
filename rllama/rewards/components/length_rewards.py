import logging
from typing import Any, Dict, Optional

from ..base import BaseReward
# Assuming the action (LLM response) is passed as a string or has a __len__ method
# Or potentially, the token count is in the info dict

logger = logging.getLogger(__name__)

class LengthPenalty(BaseReward):
    """
    Applies a penalty if the generated response exceeds a maximum length.
    Assumes the length can be determined from the 'action' or 'info' dict.
    """
    def __init__(self, name: str = "length_penalty", max_length: int = 256, penalty_per_token: float = -0.01, length_source: str = "action"):
        """
        Initializes the LengthPenalty reward component.

        Args:
            name: The name of this reward component.
            max_length: The maximum allowed length (e.g., token count).
            penalty_per_token: The penalty applied for each token exceeding max_length. Should be negative.
            length_source: Where to get the length from. Options:
                           'action': Directly use len(action). Assumes action is sequence/string.
                           'info:<key>': Look for the length in info[key]. E.g., 'info:token_count'.
        """
        super().__init__(name)
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if penalty_per_token >= 0:
            logger.warning(f"LengthPenalty '{name}' initialized with non-negative penalty_per_token ({penalty_per_token}). This will act as a bonus.")
        if not length_source.startswith("action") and not length_source.startswith("info:"):
             raise ValueError("length_source must be 'action' or 'info:<key_name>'")

        self.max_length = max_length
        self.penalty_per_token = penalty_per_token
        self.length_source = length_source
        self._info_key: Optional[str] = None
        if self.length_source.startswith("info:"):
            self._info_key = self.length_source.split(":", 1)[1]
            if not self._info_key:
                 raise ValueError("Info key cannot be empty in length_source.")

        logger.info(f"Initialized LengthPenalty '{self.name}': max_length={self.max_length}, penalty_per_token={self.penalty_per_token}, source='{self.length_source}'")

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """
        Calculates the penalty based on the length of the action/response.

        Args:
            state: The state before the action. (Not used by this component)
            action: The action taken (e.g., generated text or tokens).
            next_state: The state after the action. (Not used by this component)
            info: A dictionary containing auxiliary information.

        Returns:
            The calculated penalty (0 if within limits, negative otherwise).
        """
        current_length = -1 # Default to invalid length

        if self.length_source == "action":
            try:
                current_length = len(action)
            except TypeError:
                logger.warning(f"LengthPenalty '{self.name}': Cannot determine length from action of type {type(action)}. Returning 0 penalty.")
                return 0.0
        elif self._info_key:
            try:
                current_length = int(info.get(self._info_key, -1)) # Try to get length from info dict
            except (TypeError, ValueError):
                 logger.warning(f"LengthPenalty '{self.name}': Could not get valid integer length from info['{self._info_key}']. Returning 0 penalty.")
                 return 0.0

        if current_length < 0:
             logger.warning(f"LengthPenalty '{self.name}': Failed to determine length from source '{self.length_source}'. Returning 0 penalty.")
             return 0.0 # Failed to get length

        if current_length > self.max_length:
            excess_length = current_length - self.max_length
            penalty = excess_length * self.penalty_per_token
            # logger.debug(f"LengthPenalty '{self.name}': Length {current_length} > {self.max_length}. Applying penalty {penalty:.4f}")
            return penalty
        else:
            # logger.debug(f"LengthPenalty '{self.name}': Length {current_length} <= {self.max_length}. No penalty.")
            return 0.0 # No penalty if within limits