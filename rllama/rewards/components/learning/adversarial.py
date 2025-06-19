

import numpy as np
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from ...rewards.base import BaseReward
from ...rewards.registry import register_reward_component

@register_reward_component
class AdversarialReward(BaseReward):
    """
    Adversarial reward that intentionally challenges the agent.
    Makes the reward landscape harder as the agent improves.
    """
    
    def __init__(self,
                initial_adversity: float = 0.0,
                max_adversity: float = 1.0,
                adaptation_rate: float = 0.01,
                performance_window: int = 100,
                performance_threshold: float = 0.7):
        """
        Initialize the adversarial reward.
        
        Args:
            initial_adversity: Starting adversity level (0.0 to 1.0)
            max_adversity: Maximum adversity level
            adaptation_rate: How quickly adversity increases with performance
            performance_window: Window size for performance evaluation
            performance_threshold: Performance threshold to increase adversity
        """
        super().__init__()
        self.adversity = initial_adversity
        self.max_adversity = max_adversity
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        self.performance_threshold = performance_threshold
        
        self.performance_history = []
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate reward with adversarial adjustment.
        
        Args:
            context: Must contain 'base_reward' and 'performance' values
            
        Returns:
            Adversarially adjusted reward
        """
        base_reward = context.get('base_reward', 0.0)
        performance = context.get('performance')
        
        if performance is not None:
            # Track performance
            self.performance_history.append(performance)
            if len(self.performance_history) > self.performance_window:
                self.performance_history.pop(0)
                
            # Check if we should increase adversity
            if (len(self.performance_history) >= self.performance_window // 2 and
                np.mean(self.performance_history) >= self.performance_threshold):
                # Increase adversity
                self.adversity = min(
                    self.adversity + self.adaptation_rate,
                    self.max_adversity
                )
        
        # Apply adversarial effect: reduce reward as adversity increases
        adversarial_factor = 1.0 - self.adversity
        adjusted_reward = base_reward * adversarial_factor
        
        # Add adversity level to context for logging
        context['adversity_level'] = self.adversity
        
        return adjusted_reward
        
    def reset(self) -> None:
        """Reset performance history."""
        self.performance_history = []


@register_reward_component
class RobustnessReward(BaseReward):
    """
    Rewards robust behavior in the face of perturbations or noise.
    Helps train agents that can withstand adversarial attacks.
    """
    
    def __init__(self, 
                noise_levels: List[float] = [0.1, 0.2, 0.3],
                evaluation_frequency: int = 100,
                robustness_scale: float = 1.0):
        """
        Initialize the robustness reward.
        
        Args:
            noise_levels: Different levels of noise to test robustness against
            evaluation_frequency: How often to test robustness
            robustness_scale: Scaling factor for robustness reward
        """
        super().__init__()
        self.noise_levels = noise_levels
        self.evaluation_frequency = evaluation_frequency
        self.robustness_scale = robustness_scale
        
        self.step_count = 0
        self.last_evaluation = {}
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate robustness reward.
        
        Args:
            context: Must contain 'policy_fn' callable and 'state' value
            
        Returns:
            Reward for robustness
        """
        policy_fn = context.get('policy_fn')
        state = context.get('state')
        
        self.step_count += 1
        
        # Skip if we don't have the necessary context or not evaluation time
        if policy_fn is None or state is None:
            return 0.0
            
        # Only evaluate robustness occasionally
        if self.step_count % self.evaluation_frequency != 0:
            return 0.0
            
        # Perform robustness evaluation across noise levels
        try:
            # Get baseline action without noise
            baseline_action = policy_fn(state)
            
            robustness_scores = []
            for noise_level in self.noise_levels:
                # Apply noise to state
                noisy_state = state + np.random.normal(0, noise_level, size=state.shape)
                
                # Get action with noisy state
                noisy_action = policy_fn(noisy_state)
                
                # Calculate similarity between baseline and noisy actions
                if isinstance(baseline_action, np.ndarray) and isinstance(noisy_action, np.ndarray):
                    # For continuous actions, calculate cosine similarity
                    similarity = np.dot(baseline_action, noisy_action) / (
                        np.linalg.norm(baseline_action) * np.linalg.norm(noisy_action) + 1e-8)
                else:
                    # For discrete actions, check if they're the same
                    similarity = 1.0 if baseline_action == noisy_action else 0.0
                    
                robustness_scores.append(similarity)
                
            # Overall robustness is mean similarity across noise levels
            robustness = np.mean(robustness_scores)
            
            # Store for logging
            self.last_evaluation = {
                'step': self.step_count,
                'robustness': robustness,
                'scores': robustness_scores
            }
            
            # Return robustness reward
            return self.robustness_scale * robustness
            
        except Exception as e:
            # Fallback if evaluation fails
            return 0.0
        
    def reset(self) -> None:
        """Reset step counter and evaluation results."""
        self.step_count = 0
        self.last_evaluation = {}
