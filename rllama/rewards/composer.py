# rllama/rewards/composer.py

from typing import Dict, Any, List
import logging

from .base import BaseReward

class RewardComposer:
    """
    Composes multiple reward components into a single reward calculation.
    This class handles the execution of multiple reward components and collects
    their individual reward signals.
    """
    
    def __init__(self, components: List[BaseReward]):
        """
        Args:
            components: List of reward components to compose
        """
        self.components = components
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate rewards from all components.
        
        Args:
            context: Dictionary containing all needed context for reward calculation
                     (e.g., response, query, metadata)
                     
        Returns:
            Dictionary mapping component names to their reward values
        """
        rewards = {}
        
        for component in self.components:
            try:
                # Extract component name from class name
                component_name = component.__class__.__name__
                
                # Calculate the reward for this component
                reward = component.calculate(context)
                rewards[component_name] = reward
                
            except Exception as e:
                self.logger.error(f"Error calculating reward in {component.__class__.__name__}: {e}")
                # Add 0 reward for this component to avoid missing keys
                component_name = getattr(component, "__class__", type(component)).__name__
                rewards[component_name] = 0.0
                
        return rewards
