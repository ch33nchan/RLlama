#!/usr/bin/env python3
"""
Adaptive reward components that learn and adjust their behavior over time.
These components implement sophisticated learning mechanisms for dynamic reward shaping.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Union, List, Deque
from collections import deque, defaultdict
import time

from ...base import BaseReward
from ...registry import register_reward_component

@register_reward_component
class AdaptiveClippingReward(BaseReward):
    """
    Reward component that adaptively clips rewards based on observed statistics.
    Implements PPO-style advantage clipping with adaptive thresholds.
    """
    
    def __init__(self, 
                 base_reward_key: str = "base_reward",
                 clip_ratio: float = 0.2,
                 adaptive_clip: bool = True,
                 history_size: int = 1000,
                 percentile_threshold: float = 95.0,
                 min_clip_ratio: float = 0.05,
                 max_clip_ratio: float = 0.5,
                 adaptation_rate: float = 0.01,
                 **kwargs):
        """
        Initialize adaptive clipping reward component.
        
        Args:
            base_reward_key: Key in context containing base reward to clip
            clip_ratio: Initial clipping ratio
            adaptive_clip: Whether to adapt clipping ratio based on statistics
            history_size: Size of reward history to maintain
            percentile_threshold: Percentile for outlier detection
            min_clip_ratio: Minimum allowed clipping ratio
            max_clip_ratio: Maximum allowed clipping ratio
            adaptation_rate: Rate of adaptation for clipping ratio
        """
        super().__init__(**kwargs)
        self.base_reward_key = base_reward_key
        self.clip_ratio = clip_ratio
        self.initial_clip_ratio = clip_ratio
        self.adaptive_clip = adaptive_clip
        self.history_size = history_size
        self.percentile_threshold = percentile_threshold
        self.min_clip_ratio = min_clip_ratio
        self.max_clip_ratio = max_clip_ratio
        self.adaptation_rate = adaptation_rate
        
        # Track reward history
        self.reward_history: Deque = deque(maxlen=history_size)
        self.clipped_count = 0
        self.total_count = 0
        
        # Running statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate adaptively clipped reward."""
        # Extract base reward
        base_reward = context.get(self.base_reward_key, 0.0)
        if not isinstance(base_reward, (int, float)):
            return 0.0
            
        # Update running statistics
        self._update_statistics(base_reward)
        
        # Add to history
        self.reward_history.append(base_reward)
        
        # Calculate clipped reward
        clipped_reward = self._apply_clipping(base_reward)
        
        # Adapt clipping ratio if enabled
        if self.adaptive_clip and len(self.reward_history) >= 100:
            self._adapt_clipping_ratio()
            
        self.total_count += 1
        return clipped_reward
        
    def _update_statistics(self, reward: float) -> None:
        """Update running mean and variance using Welford's algorithm."""
        self.running_count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.running_count
        delta2 = reward - self.running_mean
        self.running_var += delta * delta2
        
    def _apply_clipping(self, reward: float) -> float:
        """Apply clipping to reward based on current statistics."""
        if self.running_count < 2:
            return reward
            
        # Calculate standard deviation
        std = math.sqrt(self.running_var / (self.running_count - 1))
        
        # Define clipping bounds based on mean and std
        clip_threshold = self.clip_ratio * std
        lower_bound = self.running_mean - clip_threshold
        upper_bound = self.running_mean + clip_threshold
        
        # Apply clipping
        if reward < lower_bound:
            self.clipped_count += 1
            return lower_bound
        elif reward > upper_bound:
            self.clipped_count += 1
            return upper_bound
        else:
            return reward
            
    def _adapt_clipping_ratio(self) -> None:
        """Adapt clipping ratio based on clipping frequency."""
        if self.total_count == 0:
            return
            
        # Calculate clipping frequency
        clip_frequency = self.clipped_count / self.total_count
        
        # Target clipping frequency (should be moderate)
        target_frequency = 0.1  # 10% of rewards should be clipped
        
        # Adjust clipping ratio based on frequency
        if clip_frequency > target_frequency:
            # Too much clipping, increase ratio
            self.clip_ratio = min(self.max_clip_ratio, 
                                self.clip_ratio * (1 + self.adaptation_rate))
        elif clip_frequency < target_frequency * 0.5:
            # Too little clipping, decrease ratio
            self.clip_ratio = max(self.min_clip_ratio,
                                self.clip_ratio * (1 - self.adaptation_rate))
                                
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            'current_clip_ratio': self.clip_ratio,
            'clip_frequency': self.clipped_count / max(1, self.total_count),
            'running_mean': self.running_mean,
            'running_std': math.sqrt(self.running_var / max(1, self.running_count - 1)),
            'total_samples': self.total_count
        }

@register_reward_component
class GradualCurriculumReward(BaseReward):
    """
    Reward component that implements curriculum learning with gradual difficulty increase.
    Starts with easier tasks and progressively increases difficulty.
    """
    
    def __init__(self,
                 difficulty_key: str = "difficulty",
                 performance_key: str = "performance", 
                 initial_difficulty: float = 0.1,
                 max_difficulty: float = 1.0,
                 success_threshold: float = 0.8,
                 failure_threshold: float = 0.4,
                 adaptation_rate: float = 0.01,
                 window_size: int = 100,
                 reward_scaling: float = 1.0,
                 **kwargs):
        """
        Initialize gradual curriculum reward component.
        
        Args:
            difficulty_key: Key in context containing current difficulty level
            performance_key: Key in context containing performance metric
            initial_difficulty: Starting difficulty level
            max_difficulty: Maximum difficulty level
            success_threshold: Performance threshold for increasing difficulty
            failure_threshold: Performance threshold for decreasing difficulty
            adaptation_rate: Rate of difficulty adaptation
            window_size: Size of performance history window
            reward_scaling: Scaling factor for final reward
        """
        super().__init__(**kwargs)
        self.difficulty_key = difficulty_key
        self.performance_key = performance_key
        self.current_difficulty = initial_difficulty
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        self.failure_threshold = failure_threshold
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # Track performance history
        self.performance_history: Deque = deque(maxlen=window_size)
        self.step_count = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate curriculum-based reward."""
        # Extract performance and difficulty
        performance = context.get(self.performance_key, 0.0)
        difficulty = context.get(self.difficulty_key, self.current_difficulty)
        
        if not isinstance(performance, (int, float)):
            performance = 0.0
            
        # Update performance history
        self.performance_history.append(performance)
        self.step_count += 1
        
        # Adapt difficulty based on recent performance
        if len(self.performance_history) >= self.window_size:
            self._adapt_difficulty()
            
        # Calculate reward based on performance and difficulty
        base_reward = performance * difficulty
        
        # Bonus for maintaining performance at higher difficulties
        difficulty_bonus = (self.current_difficulty - self.initial_difficulty) * 0.1
        
        # Final reward
        reward = (base_reward + difficulty_bonus) * self.reward_scaling
        
        return reward
        
    def _adapt_difficulty(self) -> None:
        """Adapt difficulty based on recent performance."""
        if not self.performance_history:
            return
            
        # Calculate recent average performance
        recent_performance = np.mean(list(self.performance_history))
        
        # Increase difficulty if performing well
        if recent_performance >= self.success_threshold:
            difficulty_increase = self.adaptation_rate * (recent_performance - self.success_threshold)
            self.current_difficulty = min(self.max_difficulty, 
                                        self.current_difficulty + difficulty_increase)
                                        
        # Decrease difficulty if performing poorly
        elif recent_performance <= self.failure_threshold:
            difficulty_decrease = self.adaptation_rate * (self.failure_threshold - recent_performance)
            self.current_difficulty = max(self.initial_difficulty,
                                        self.current_difficulty - difficulty_decrease)
                                        
    def get_current_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
        
    def reset_difficulty(self) -> None:
        """Reset difficulty to initial level."""
        self.current_difficulty = self.initial_difficulty
        self.performance_history.clear()
        self.step_count = 0

@register_reward_component
class AdaptiveNormalizationReward(BaseReward):
    """
    Reward component that adaptively normalizes rewards based on running statistics.
    Implements various normalization strategies with online adaptation.
    """
    
    def __init__(self,
                 reward_key: str = "raw_reward",
                 normalization_type: str = "z_score",
                 adaptation_rate: float = 0.01,
                 min_std: float = 1e-6,
                 clip_range: Optional[float] = 3.0,
                 momentum: float = 0.99,
                 **kwargs):
        """
        Initialize adaptive normalization reward component.
        
        Args:
            reward_key: Key in context containing raw reward to normalize
            normalization_type: Type of normalization ("z_score", "min_max", "robust")
            adaptation_rate: Rate of adaptation for statistics
            min_std: Minimum standard deviation to prevent division by zero
            clip_range: Range to clip normalized rewards (if not None)
            momentum: Momentum for exponential moving average
        """
        super().__init__(**kwargs)
        self.reward_key = reward_key
        self.normalization_type = normalization_type
        self.adaptation_rate = adaptation_rate
        self.min_std = min_std
        self.clip_range = clip_range
        self.momentum = momentum
        
        # Running statistics
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_min = float('inf')
        self.running_max = float('-inf')
        self.running_median = 0.0
        self.running_mad = 1.0  # Median Absolute Deviation
        self.count = 0
        
        # For robust statistics
        self.recent_values: Deque = deque(maxlen=1000)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate adaptively normalized reward."""
        # Extract raw reward
        raw_reward = context.get(self.reward_key, 0.0)
        if not isinstance(raw_reward, (int, float)):
            return 0.0
            
        # Update statistics
        self._update_statistics(raw_reward)
        
        # Apply normalization
        if self.normalization_type == "z_score":
            normalized = self._z_score_normalize(raw_reward)
        elif self.normalization_type == "min_max":
            normalized = self._min_max_normalize(raw_reward)
        elif self.normalization_type == "robust":
            normalized = self._robust_normalize(raw_reward)
        else:
            normalized = raw_reward
            
        # Apply clipping if specified
        if self.clip_range is not None:
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)
            
        return normalized
        
    def _update_statistics(self, reward: float) -> None:
        """Update running statistics."""
        self.count += 1
        
        # Update mean and variance with exponential moving average
        if self.count == 1:
            self.running_mean = reward
            self.running_var = 1.0
        else:
            # Exponential moving average
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * reward
            
            # Update variance
            diff = reward - self.running_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * diff * diff
            
        # Update min/max
        self.running_min = min(self.running_min, reward)
        self.running_max = max(self.running_max, reward)
        
        # Update recent values for robust statistics
        self.recent_values.append(reward)
        
        # Update robust statistics periodically
        if len(self.recent_values) >= 50 and self.count % 10 == 0:
            self._update_robust_statistics()
            
    def _update_robust_statistics(self) -> None:
        """Update robust statistics (median, MAD)."""
        if not self.recent_values:
            return
            
        values = list(self.recent_values)
        self.running_median = np.median(values)
        
        # Calculate Median Absolute Deviation
        mad = np.median(np.abs(values - self.running_median))
        self.running_mad = max(mad, self.min_std)
        
    def _z_score_normalize(self, reward: float) -> float:
        """Apply z-score normalization."""
        std = math.sqrt(max(self.running_var, self.min_std))
        return (reward - self.running_mean) / std
        
    def _min_max_normalize(self, reward: float) -> float:
        """Apply min-max normalization."""
        if self.running_max <= self.running_min:
            return 0.0
        return (reward - self.running_min) / (self.running_max - self.running_min)
        
    def _robust_normalize(self, reward: float) -> float:
        """Apply robust normalization using median and MAD."""
        return (reward - self.running_median) / self.running_mad
        
    def get_statistics(self) -> Dict[str, float]:
        """Get current normalization statistics."""
        return {
            'running_mean': self.running_mean,
            'running_std': math.sqrt(self.running_var),
            'running_min': self.running_min,
            'running_max': self.running_max,
            'running_median': self.running_median,
            'running_mad': self.running_mad,
            'count': self.count
        }
