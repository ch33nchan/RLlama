

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from ...rewards.base import BaseReward
from ...rewards.registry import register_reward_component

@register_reward_component
class AdaptiveClippingReward(BaseReward):
    """
    Reward component that adaptively clips rewards based on recent history.
    Helps prevent reward exploitation and maintain stability.
    """
    
    def __init__(self, 
                window_size: int = 100, 
                clip_percentile: float = 95.0,
                minimum_window: int = 10):
        """
        Initialize the adaptive clipping reward.
        
        Args:
            window_size: Size of the history window to consider
            clip_percentile: Percentile for clipping (e.g., 95.0 clips above 95th percentile)
            minimum_window: Minimum samples required before clipping takes effect
        """
        super().__init__()
        self.window_size = window_size
        self.clip_percentile = clip_percentile
        self.minimum_window = minimum_window
        self.reward_history = []
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Apply adaptive clipping to the input reward.
        
        Args:
            context: Must contain 'reward' key with the raw reward to clip
            
        Returns:
            Clipped reward value
        """
        raw_reward = context.get('reward')
        
        if raw_reward is None:
            return 0.0
            
        # Add to history
        self.reward_history.append(raw_reward)
        
        # Maintain window size
        if len(self.reward_history) > self.window_size:
            self.reward_history = self.reward_history[-self.window_size:]
            
        # Apply clipping if we have enough samples
        if len(self.reward_history) >= self.minimum_window:
            clip_value = np.percentile(self.reward_history, self.clip_percentile)
            return min(raw_reward, clip_value)
        else:
            return raw_reward
            
    def reset(self) -> None:
        """Reset reward history."""
        self.reward_history = []


@register_reward_component
class GradualCurriculumReward(BaseReward):
    """
    Implements a curriculum that gradually increases task difficulty.
    Adjusts rewards based on agent's recent performance.
    """
    
    def __init__(self, 
                 initial_difficulty: float = 0.1,
                 max_difficulty: float = 1.0, 
                 difficulty_step: float = 0.05,
                 success_threshold: float = 0.8,
                 window_size: int = 100):
        """
        Initialize the curriculum reward.
        
        Args:
            initial_difficulty: Starting difficulty level (0.0 to 1.0)
            max_difficulty: Maximum difficulty level
            difficulty_step: How much to increase difficulty on success
            success_threshold: Performance threshold to increase difficulty
            window_size: Window size for performance evaluation
        """
        super().__init__()
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.difficulty_step = difficulty_step
        self.success_threshold = success_threshold
        self.window_size = window_size
        
        self.success_history = []
        self.step_count = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate curriculum-adjusted reward.
        
        Args:
            context: Must contain 'success' boolean and 'base_reward' float
            
        Returns:
            Curriculum-adjusted reward
        """
        success = context.get('success', False)
        base_reward = context.get('base_reward', 0.0)
        
        # Track success
        self.success_history.append(1.0 if success else 0.0)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
            
        # Check if we should increase difficulty
        if (len(self.success_history) >= self.window_size // 2 and
            np.mean(self.success_history) >= self.success_threshold):
            # Increase difficulty
            self.current_difficulty = min(
                self.current_difficulty + self.difficulty_step,
                self.max_difficulty
            )
            # Reset history after increasing difficulty
            self.success_history = []
            
        # Scale reward based on current difficulty
        difficulty_factor = 1.0 + self.current_difficulty  # Higher difficulty, higher reward
        scaled_reward = base_reward * difficulty_factor
        
        # Include difficulty level in context for logging
        context['curriculum_difficulty'] = self.current_difficulty
        
        self.step_count += 1
        return scaled_reward
        
    def reset(self) -> None:
        """Reset success history."""
        self.success_history = []
        self.step_count = 0


@register_reward_component
class UncertaintyBasedReward(BaseReward):
    """
    Rewards actions based on model uncertainty reduction.
    Encourages exploring uncertain areas to improve model quality.
    """
    
    def __init__(self, 
                exploration_weight: float = 1.0,
                decay_factor: float = 0.999,
                min_uncertainty: float = 0.1):
        """
        Initialize the uncertainty-based reward.
        
        Args:
            exploration_weight: Weight for exploration vs exploitation
            decay_factor: How quickly exploration importance decays
            min_uncertainty: Minimum uncertainty to consider
        """
        super().__init__()
        self.exploration_weight = exploration_weight
        self.decay_factor = decay_factor
        self.min_uncertainty = min_uncertainty
        self.step = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate reward based on uncertainty reduction.
        
        Args:
            context: Must contain 'prev_uncertainty' and 'curr_uncertainty' floats
            
        Returns:
            Reward for uncertainty reduction
        """
        prev_uncertainty = context.get('prev_uncertainty')
        curr_uncertainty = context.get('curr_uncertainty')
        
        if prev_uncertainty is None or curr_uncertainty is None:
            return 0.0
            
        # Calculate uncertainty reduction
        uncertainty_reduction = max(0.0, prev_uncertainty - curr_uncertainty)
        
        # Apply decay to exploration weight
        current_weight = self.exploration_weight * (self.decay_factor ** self.step)
        
        # Calculate reward
        reward = current_weight * uncertainty_reduction
        
        self.step += 1
        return reward
        
    def reset(self) -> None:
        """Reset step counter."""
        self.step = 0
