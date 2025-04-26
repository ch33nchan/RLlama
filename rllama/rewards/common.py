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


class HolePenaltyReward(BaseReward):
    """Applies a penalty if the agent lands on a hole state."""

    def __init__(self, penalty: float = -1.0, hole_char: str = 'H'):
        """
        Args:
            penalty: The negative reward value to assign for falling into a hole.
            hole_char: The character representing a hole in the environment's description (used if info doesn't directly indicate hole).
        """
        if penalty > 0:
            print(f"Warning: HolePenaltyReward initialized with positive penalty ({penalty}). Usually should be negative.")
        self.penalty = penalty
        self.hole_char = hole_char
        print(f"Initialized HolePenaltyReward (Penalty: {self.penalty})")


    @property
    def name(self) -> str:
        return "hole_penalty"

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """
        Checks if the next state corresponds to a hole.

        Relies on the environment providing information about the outcome,
        either directly in `info` or by checking the map description if available.
        """
        # Option 1: Check if info explicitly tells us we landed in a hole
        # (This depends on the specific environment wrapper or info dict structure)
        if info.get("landed_on_hole", False): # Example key, adjust if needed
             return self.penalty

        # Option 2: Check the environment description if available (less reliable)
        # This requires access to the environment's map/description
        env_desc = info.get("env_desc") # Assuming env description is passed in info
        if env_desc is not None and hasattr(env_desc, 'shape') and len(env_desc.shape) == 2:
             try:
                 rows, cols = env_desc.shape
                 row, col = divmod(next_state, cols) # Assumes discrete state space maps to grid
                 if 0 <= row < rows and 0 <= col < cols:
                     if env_desc[row, col].decode('utf-8') == self.hole_char:
                         return self.penalty
             except Exception:
                 # Handle cases where state doesn't map cleanly or desc is wrong format
                 pass # Fall through to default reward

        # Option 3: Check termination condition if it implies falling in a hole
        # This is tricky as termination can also mean reaching the goal
        terminated = info.get("terminated", False)
        reached_goal = info.get("reached_goal", False) # Need a way to distinguish goal termination
        if terminated and not reached_goal:
             # Assume termination without reaching goal means falling in hole (specific to FrozenLake)
             # This is fragile and depends heavily on how termination/goal flags are set.
             # A more robust approach uses env-specific info (like Option 1 or 2).
             # Let's rely on the demo script passing specific info if possible.
             pass # Avoid using this fragile logic for now.


        return 0.0 # No penalty if not identified as a hole