from typing import List, Dict, Any, Optional, Tuple # Ensure Tuple is imported
import numpy as np
from .base import BaseReward # Make sure BaseReward is correctly imported
from .normalization import Normalizer # CRITICAL: Imports the Normalizer from the cleaned file
import logging

logger = logging.getLogger(__name__) # Assuming you have logging setup

class RewardComposer:
    """
    Combines rewards from multiple components, applies normalization,
    and provides a composite reward.
    """
    def __init__(self,
                 components: List[BaseReward],
                 normalization_strategy: str = 'none', # Accepts this
                 norm_window: int = 100,               # Accepts this
                 epsilon: float = 1e-8,                # Accepts this
                 clip_range: Optional[tuple] = None,   # Accepts this
                 **kwargs):                             # Catches any other unexpected args
        self.components = components
        self.component_names = [comp.name for comp in components]
        
        self.normalization_strategy = normalization_strategy
        self.norm_window = norm_window
        self.epsilon = epsilon
        self.clip_range = clip_range # Store clip_range if provided

        self.normalizers: Dict[str, Normalizer] = {}
        if self.normalization_strategy != 'none':
            for name in self.component_names:
                # Instantiates the Normalizer class from normalization.py
                self.normalizers[name] = Normalizer(
                    strategy=self.normalization_strategy,
                    window_size=self.norm_window,
                    epsilon=self.epsilon
                )
        
        logger.info(f"RewardComposer initialized with components: {self.component_names}")
        logger.info(f"Normalization strategy: {self.normalization_strategy}, window: {self.norm_window}, epsilon: {self.epsilon}")
        if self.clip_range:
            logger.info(f"Reward clipping range: {self.clip_range}")
        if kwargs:
            logger.warning(f"RewardComposer received unexpected keyword arguments: {kwargs}")


    def compute_rewards(self, prompts: List[str], responses: List[str], **kwargs) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
        """
        Computes rewards from all components for a batch of prompts and responses.

        Args:
            prompts: A list of prompt strings.
            responses: A list of corresponding response strings.
            **kwargs: Additional context to pass to reward components.

        Returns:
            A tuple containing:
            - A list of dictionaries, where each dictionary maps component names to their raw (pre-normalized) reward values.
            - A list of dictionaries, where each dictionary maps component names to their detailed info/metadata.
        """
        batch_size = len(prompts)
        all_raw_rewards: List[Dict[str, float]] = [{} for _ in range(batch_size)]
        all_detailed_infos: List[Dict[str, Any]] = [{} for _ in range(batch_size)]

        for component in self.components:
            try:
                # Assuming component.compute_reward returns a list of (reward_value, detailed_info_dict) tuples
                # or just a list of reward_values if detailed_info is not supported by that component.
                component_outputs = component.compute_reward(prompts, responses, **kwargs)

                for i in range(batch_size):
                    if isinstance(component_outputs[i], tuple) and len(component_outputs[i]) == 2:
                        reward_value, detailed_info = component_outputs[i]
                    else: # Assuming it's just the reward value
                        reward_value = component_outputs[i]
                        detailed_info = {} # No detailed info provided by this component

                    all_raw_rewards[i][component.name] = float(reward_value)
                    all_detailed_infos[i][component.name] = detailed_info
                    all_detailed_infos[i].update({ # Add prompt and response for context if not already there
                        f"{component.name}_prompt": prompts[i],
                        f"{component.name}_response": responses[i]
                    })


            except Exception as e:
                logger.error(f"Error computing reward for component {component.name}: {e}", exc_info=True)
                for i in range(batch_size): # Ensure structure is maintained even on error
                    all_raw_rewards[i][component.name] = 0.0 # Default to 0 on error
                    all_detailed_infos[i][component.name] = {"error": str(e)}

        return all_raw_rewards, all_detailed_infos

    def normalize_rewards(self, batch_raw_rewards: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Normalizes the raw rewards from components if a normalization strategy is set.
        Updates internal normalizer states.
        """
        if self.normalization_strategy == 'none':
            return batch_raw_rewards

        batch_normalized_rewards: List[Dict[str, float]] = [{} for _ in range(len(batch_raw_rewards))]

        # Collect all rewards for each component across the batch to update normalizers
        for comp_name in self.component_names:
            if comp_name in self.normalizers:
                rewards_for_this_component = [raw_rewards_dict.get(comp_name, 0.0) for raw_rewards_dict in batch_raw_rewards]
                self.normalizers[comp_name].update(rewards_for_this_component) # Update normalizer state

        # Now apply normalization using the updated normalizers
        for i, raw_rewards_dict in enumerate(batch_raw_rewards):
            for comp_name, raw_reward_val in raw_rewards_dict.items():
                if comp_name in self.normalizers:
                    normalized_val = self.normalizers[comp_name].normalize_value(raw_reward_val)
                    batch_normalized_rewards[i][comp_name] = normalized_val
                else: # Should not happen if normalizers are created for all components
                    batch_normalized_rewards[i][comp_name] = raw_reward_val
        
        return batch_normalized_rewards

    def reset_normalizers(self):
        """Resets the state of all normalizers."""
        for normalizer in self.normalizers.values():
            normalizer.reset()
        logger.info("RewardComposer normalizers have been reset.")

    # You might need other methods depending on how RewardComposer is used,
    # e.g., a method to get the current state of normalizers or to apply shaping.