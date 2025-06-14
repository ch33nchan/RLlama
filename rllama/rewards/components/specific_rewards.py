from typing import Any, Dict
from .base import BaseReward
from .registry import reward_registry

# Pass the class name as a string argument to the decorator
@reward_registry.register("ToxicityPenalty")
class ToxicityPenalty(BaseReward):
    """
    A reward component that applies a fixed penalty if a toxicity score
    exceeds a given threshold.
    """
    def __init__(self, name: str, penalty_value: float, threshold: float, **kwargs: Any):
        """
        Initializes the ToxicityPenalty reward component.

        Args:
            name: The name of the reward component.
            penalty_value: The negative reward value to apply if threshold is exceeded.
            threshold: The toxicity score threshold.
            **kwargs: Additional keyword arguments (ignored by this component but captured for flexibility).
        """
        super().__init__(name)
        if penalty_value > 0:
            print(f"Warning: ToxicityPenalty '{name}' initialized with a positive penalty_value ({penalty_value}). It's typically negative.")
        self.penalty_value = penalty_value
        self.threshold = threshold
        print(f"Initialized ToxicityPenalty '{name}' with threshold={threshold}, penalty={penalty_value}") # Debug print

    @property # Add concrete implementation for the name property
    def name(self) -> str:
        """Returns the name of the reward component."""
        return self._name # Assumes BaseReward.__init__ sets self._name

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """
        Calculates the toxicity penalty.

        Assumes the toxicity score is provided in the `info` dictionary
        under the key 'toxicity_score'.

        Args:
            state: The current state (unused).
            action: The action taken (unused).
            next_state: The next state (unused).
            info: A dictionary potentially containing 'toxicity_score'.

        Returns:
            The penalty value if toxicity exceeds the threshold, otherwise 0.0.
        """
        toxicity_score = info.get('toxicity_score')

        if toxicity_score is None:
            # print(f"Warning: 'toxicity_score' not found in info dict for reward '{self.name}'. Returning 0.0 reward.")
            return 0.0 # Or raise an error, depending on desired behavior

        if not isinstance(toxicity_score, (float, int)):
             print(f"Warning: 'toxicity_score' in info dict for reward '{self.name}' is not a number ({type(toxicity_score)}). Returning 0.0 reward.")
             return 0.0

        if toxicity_score > self.threshold:
            # print(f"Debug: Toxicity {toxicity_score} > {self.threshold}. Applying penalty {self.penalty_value} for '{self.name}'.") # Debug print
            return self.penalty_value
        else:
            # print(f"Debug: Toxicity {toxicity_score} <= {self.threshold}. No penalty for '{self.name}'.") # Debug print
            return 0.0


# Pass the class name as a string argument to the decorator
@reward_registry.register("PreferenceScoreReward")
class PreferenceScoreReward(BaseReward):
    """
    A reward component that uses a preference model to score
    a generated response or state-action pair.
    """
    def __init__(self, name: str, model_path: str, **kwargs: Any):
        """
        Initializes the PreferenceScoreReward component.

        Args:
            name: The name of the reward component.
            model_path: Path to the preference model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name)
        self.model_path = model_path
        # TODO: Implement actual model loading here based on model_path
        # self.preference_model = self._load_model(model_path)
        print(f"Initialized PreferenceScoreReward '{name}' with model_path='{model_path}' (Model loading not implemented yet)") # Debug print

    @property # Add concrete implementation for the name property
    def name(self) -> str:
        """Returns the name of the reward component."""
        return self._name # Assumes BaseReward.__init__ sets self._name

    # def _load_model(self, path: str):
    #     # Placeholder for model loading logic (e.g., using transformers, torch, etc.)
    #     print(f"Placeholder: Would load model from {path}")
    #     return None # Return the actual loaded model object

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """
        Calculates the preference score reward.

        This is a placeholder implementation. The actual implementation would
        use the loaded preference model to score the relevant input (e.g.,
        the generated text in 'action' or 'next_state').

        Args:
            state: The current state.
            action: The action taken (e.g., generated text).
            next_state: The next state.
            info: Additional information dictionary.

        Returns:
            A placeholder score (0.0). Replace with actual model inference.
        """
        # TODO: Implement actual inference using self.preference_model
        # Example: score = self.preference_model.predict(action)
        score = 0.0 # Placeholder
        # print(f"Debug: PreferenceScoreReward '{self.name}' called. Returning placeholder score: {score}") # Debug print
        return score