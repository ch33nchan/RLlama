#!/usr/bin/env python3
"""
Meta-learning reward components that adapt and learn from experience.
These components implement sophisticated meta-learning mechanisms for reward optimization.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Union, List, Deque, Callable, Tuple
from collections import deque, defaultdict
import time
import copy

from ...base import BaseReward
from ...registry import register_reward_component

@register_reward_component
class MetaLearningReward(BaseReward):
    """
    Meta-learning reward component that learns to adapt its parameters based on task performance.
    Implements Model-Agnostic Meta-Learning (MAML) style adaptation for reward functions.
    """
    
    def __init__(self,
                 base_reward_key: str = "base_reward",
                 meta_learning_rate: float = 0.01,
                 adaptation_steps: int = 5,
                 task_window_size: int = 100,
                 parameter_keys: List[str] = None,
                 adaptation_threshold: float = 0.1,
                 **kwargs):
        """
        Initialize meta-learning reward component.
        
        Args:
            base_reward_key: Key in context containing base reward to adapt
            meta_learning_rate: Learning rate for meta-parameter updates
            adaptation_steps: Number of gradient steps for task adaptation
            task_window_size: Size of window for task performance tracking
            parameter_keys: List of parameter keys to adapt
            adaptation_threshold: Threshold for triggering adaptation
        """
        super().__init__(**kwargs)
        self.base_reward_key = base_reward_key
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_steps = adaptation_steps
        self.task_window_size = task_window_size
        self.parameter_keys = parameter_keys or ["weight", "scaling", "threshold"]
        self.adaptation_threshold = adaptation_threshold
        
        # Meta-parameters (learnable)
        self.meta_parameters = {
            "weight": 1.0,
            "scaling": 1.0,
            "threshold": 0.5,
            "bias": 0.0
        }
        
        # Task-specific adapted parameters
        self.adapted_parameters = self.meta_parameters.copy()
        
        # Performance tracking
        self.task_performance_history: Deque = deque(maxlen=task_window_size)
        self.parameter_gradients = {key: 0.0 for key in self.meta_parameters}
        
        # Task detection
        self.current_task_id = 0
        self.task_change_detector = TaskChangeDetector()
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate meta-learned reward."""
        # Extract base reward
        base_reward = context.get(self.base_reward_key, 0.0)
        if not isinstance(base_reward, (int, float)):
            return 0.0
            
        # Detect task changes
        task_changed = self._detect_task_change(context)
        if task_changed:
            self._adapt_to_new_task(context)
            
        # Apply current adapted parameters
        adapted_reward = self._apply_adaptation(base_reward, context)
        
        # Track performance for meta-learning
        self._track_performance(adapted_reward, context)
        
        # Update meta-parameters periodically
        if len(self.task_performance_history) >= self.task_window_size:
            self._update_meta_parameters()
            
        return adapted_reward
        
    def _detect_task_change(self, context: Dict[str, Any]) -> bool:
        """Detect if the task has changed."""
        return self.task_change_detector.detect_change(context)
        
    def _adapt_to_new_task(self, context: Dict[str, Any]) -> None:
        """Adapt parameters for a new task using few-shot learning."""
        self.current_task_id += 1
        
        # Reset adapted parameters to meta-parameters
        self.adapted_parameters = self.meta_parameters.copy()
        
        # Perform few-shot adaptation if we have recent data
        if len(self.task_performance_history) > 5:
            self._few_shot_adaptation()
            
    def _few_shot_adaptation(self) -> None:
        """Perform few-shot adaptation using recent performance data."""
        # Get recent performance data
        recent_performance = list(self.task_performance_history)[-10:]
        
        if len(recent_performance) < 3:
            return
            
        # Simple gradient-based adaptation
        for step in range(self.adaptation_steps):
            # Compute gradients based on recent performance
            gradients = self._compute_adaptation_gradients(recent_performance)
            
            # Update adapted parameters
            for key in self.parameter_keys:
                if key in self.adapted_parameters:
                    gradient = gradients.get(key, 0.0)
                    self.adapted_parameters[key] += self.meta_learning_rate * gradient
                    
    def _compute_adaptation_gradients(self, performance_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute gradients for parameter adaptation."""
        gradients = {}
        
        if len(performance_data) < 2:
            return gradients
            
        # Simple finite difference approximation
        for key in self.parameter_keys:
            if key in self.adapted_parameters:
                # Estimate gradient using performance trend
                recent_rewards = [p.get('reward', 0.0) for p in performance_data]
                
                if len(recent_rewards) >= 2:
                    # Compute trend
                    trend = np.mean(np.diff(recent_rewards))
                    
                    # Gradient is proportional to negative trend (want to increase reward)
                    gradients[key] = -trend * 0.1  # Scale factor
                    
        return gradients
        
    def _apply_adaptation(self, base_reward: float, context: Dict[str, Any]) -> float:
        """Apply adapted parameters to base reward."""
        # Apply scaling and bias
        adapted_reward = base_reward * self.adapted_parameters.get("scaling", 1.0)
        adapted_reward += self.adapted_parameters.get("bias", 0.0)
        
        # Apply weight
        adapted_reward *= self.adapted_parameters.get("weight", 1.0)
        
        # Apply threshold if specified
        threshold = self.adapted_parameters.get("threshold", None)
        if threshold is not None:
            if adapted_reward < threshold:
                adapted_reward *= 0.5  # Penalty for below threshold
                
        return adapted_reward
        
    def _track_performance(self, reward: float, context: Dict[str, Any]) -> None:
        """Track performance for meta-learning."""
        performance_entry = {
            'reward': reward,
            'timestamp': time.time(),
            'task_id': self.current_task_id,
            'parameters': self.adapted_parameters.copy()
        }
        
        # Add context-specific metrics if available
        if 'performance_metric' in context:
            performance_entry['metric'] = context['performance_metric']
            
        self.task_performance_history.append(performance_entry)
        
    def _update_meta_parameters(self) -> None:
        """Update meta-parameters based on accumulated experience."""
        # Analyze performance across different tasks
        task_performances = defaultdict(list)
        
        for entry in self.task_performance_history:
            task_id = entry['task_id']
            task_performances[task_id].append(entry)
            
        # Compute meta-gradients
        meta_gradients = self._compute_meta_gradients(task_performances)
        
        # Update meta-parameters
        for key, gradient in meta_gradients.items():
            if key in self.meta_parameters:
                self.meta_parameters[key] += self.meta_learning_rate * gradient
                
        # Clip meta-parameters to reasonable ranges
        self._clip_meta_parameters()
        
    def _compute_meta_gradients(self, task_performances: Dict[int, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Compute meta-gradients across tasks."""
        meta_gradients = {}
        
        # For each parameter, compute how changes affect average performance
        for param_key in self.parameter_keys:
            if param_key not in self.meta_parameters:
                continue
                
            gradient_sum = 0.0
            count = 0
            
            for task_id, performances in task_performances.items():
                if len(performances) < 2:
                    continue
                    
                # Analyze parameter-performance relationship
                param_values = [p['parameters'].get(param_key, 0.0) for p in performances]
                rewards = [p['reward'] for p in performances]
                
                if len(param_values) >= 2 and len(rewards) >= 2:
                    # Simple correlation-based gradient
                    correlation = np.corrcoef(param_values, rewards)[0, 1]
                    if not np.isnan(correlation):
                        gradient_sum += correlation
                        count += 1
                        
            if count > 0:
                meta_gradients[param_key] = gradient_sum / count
            else:
                meta_gradients[param_key] = 0.0
                
        return meta_gradients
        
    def _clip_meta_parameters(self) -> None:
        """Clip meta-parameters to reasonable ranges."""
        # Define reasonable ranges for each parameter
        param_ranges = {
            "weight": (0.1, 10.0),
            "scaling": (0.1, 5.0),
            "threshold": (-2.0, 2.0),
            "bias": (-1.0, 1.0)
        }
        
        for key, (min_val, max_val) in param_ranges.items():
            if key in self.meta_parameters:
                self.meta_parameters[key] = np.clip(
                    self.meta_parameters[key], min_val, max_val
                )
                
    def get_meta_state(self) -> Dict[str, Any]:
        """Get current meta-learning state."""
        return {
            'meta_parameters': self.meta_parameters.copy(),
            'adapted_parameters': self.adapted_parameters.copy(),
            'current_task_id': self.current_task_id,
            'performance_history_length': len(self.task_performance_history)
        }

class TaskChangeDetector:
    """Detects when the task/environment has changed."""
    
    def __init__(self, window_size: int = 20, threshold: float = 0.3):
        """
        Initialize task change detector.
        
        Args:
            window_size: Size of window for change detection
            threshold: Threshold for detecting significant changes
        """
        self.window_size = window_size
        self.threshold = threshold
        self.feature_history: Deque = deque(maxlen=window_size)
        
    def detect_change(self, context: Dict[str, Any]) -> bool:
        """Detect if task has changed based on context features."""
        # Extract features from context
        features = self._extract_features(context)
        
        if features is None:
            return False
            
        # Add to history
        self.feature_history.append(features)
        
        # Need sufficient history for change detection
        if len(self.feature_history) < self.window_size:
            return False
            
        # Detect change using distribution shift
        return self._detect_distribution_shift()
        
    def _extract_features(self, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract numerical features from context."""
        features = []
        
        # Look for numerical values in context
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                features.extend([float(x) for x in value])
                
        if features:
            return np.array(features)
        else:
            return None
            
    def _detect_distribution_shift(self) -> bool:
        """Detect distribution shift in features."""
        if len(self.feature_history) < self.window_size:
            return False
            
        # Split history into two halves
        mid_point = len(self.feature_history) // 2
        first_half = list(self.feature_history)[:mid_point]
        second_half = list(self.feature_history)[mid_point:]
        
        if len(first_half) < 3 or len(second_half) < 3:
            return False
            
        # Convert to arrays
        first_half_array = np.array(first_half)
        second_half_array = np.array(second_half)
        
        # Ensure same feature dimensionality
        if first_half_array.shape[1] != second_half_array.shape[1]:
            return False
            
        # Compute means and check for significant difference
        mean1 = np.mean(first_half_array, axis=0)
        mean2 = np.mean(second_half_array, axis=0)
        
        # Compute normalized difference
        diff = np.linalg.norm(mean1 - mean2)
        norm_factor = np.linalg.norm(mean1) + np.linalg.norm(mean2) + 1e-8
        
        normalized_diff = diff / norm_factor
        
        return normalized_diff > self.threshold

@register_reward_component
class UncertaintyBasedReward(BaseReward):
    """
    Reward component that incorporates uncertainty estimates for exploration.
    Provides higher rewards for uncertain states to encourage exploration.
    """
    
    def __init__(self,
                 uncertainty_key: str = "uncertainty",
                 base_reward_key: str = "base_reward",
                 uncertainty_weight: float = 0.1,
                 uncertainty_type: str = "additive",
                 exploration_bonus: float = 0.05,
                 confidence_threshold: float = 0.8,
                 **kwargs):
        """
        Initialize uncertainty-based reward component.
        
        Args:
            uncertainty_key: Key in context containing uncertainty estimate
            base_reward_key: Key in context containing base reward
            uncertainty_weight: Weight for uncertainty in final reward
            uncertainty_type: How to combine uncertainty ("additive", "multiplicative", "bonus")
            exploration_bonus: Bonus for high uncertainty states
            confidence_threshold: Threshold for high confidence states
        """
        super().__init__(**kwargs)
        self.uncertainty_key = uncertainty_key
        self.base_reward_key = base_reward_key
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_type = uncertainty_type
        self.exploration_bonus = exploration_bonus
        self.confidence_threshold = confidence_threshold
        
        # Track uncertainty statistics
        self.uncertainty_history: Deque = deque(maxlen=1000)
        self.running_uncertainty_mean = 0.0
        self.running_uncertainty_std = 1.0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate uncertainty-based reward."""
        # Extract base reward and uncertainty
        base_reward = context.get(self.base_reward_key, 0.0)
        uncertainty = context.get(self.uncertainty_key, 0.0)
        
        if not isinstance(base_reward, (int, float)):
            base_reward = 0.0
        if not isinstance(uncertainty, (int, float)):
            uncertainty = 0.0
            
        # Update uncertainty statistics
        self._update_uncertainty_stats(uncertainty)
        
        # Normalize uncertainty
        normalized_uncertainty = self._normalize_uncertainty(uncertainty)
        
        # Apply uncertainty-based modification
        if self.uncertainty_type == "additive":
            final_reward = base_reward + self.uncertainty_weight * normalized_uncertainty
        elif self.uncertainty_type == "multiplicative":
            uncertainty_factor = 1.0 + self.uncertainty_weight * normalized_uncertainty
            final_reward = base_reward * uncertainty_factor
        elif self.uncertainty_type == "bonus":
            exploration_bonus = self._compute_exploration_bonus(normalized_uncertainty)
            final_reward = base_reward + exploration_bonus
        else:
            final_reward = base_reward
            
        return final_reward
        
    def _update_uncertainty_stats(self, uncertainty: float) -> None:
        """Update running statistics for uncertainty."""
        self.uncertainty_history.append(uncertainty)
        
        if len(self.uncertainty_history) > 10:
            # Update running mean and std
            recent_uncertainties = list(self.uncertainty_history)[-100:]
            self.running_uncertainty_mean = np.mean(recent_uncertainties)
            self.running_uncertainty_std = np.std(recent_uncertainties) + 1e-8
            
    def _normalize_uncertainty(self, uncertainty: float) -> float:
        """Normalize uncertainty using running statistics."""
        if self.running_uncertainty_std == 0:
            return 0.0
            
        # Z-score normalization
        normalized = (uncertainty - self.running_uncertainty_mean) / self.running_uncertainty_std
        
        # Clip to reasonable range
        return np.clip(normalized, -3.0, 3.0)
        
    def _compute_exploration_bonus(self, normalized_uncertainty: float) -> float:
        """Compute exploration bonus based on uncertainty."""
        # Higher bonus for higher uncertainty
        if normalized_uncertainty > 1.0:  # High uncertainty
            bonus = self.exploration_bonus * (1.0 + normalized_uncertainty)
        elif normalized_uncertainty < -1.0:  # Very low uncertainty (high confidence)
            # Small penalty for very confident predictions
            bonus = -self.exploration_bonus * 0.1
        else:
            # Moderate uncertainty gets moderate bonus
            bonus = self.exploration_bonus * normalized_uncertainty * 0.5
            
        return bonus
        
    def get_uncertainty_stats(self) -> Dict[str, float]:
        """Get current uncertainty statistics."""
        return {
            'running_mean': self.running_uncertainty_mean,
            'running_std': self.running_uncertainty_std,
            'history_length': len(self.uncertainty_history),
            'recent_uncertainty': self.uncertainty_history[-1] if self.uncertainty_history else 0.0
        }

@register_reward_component
class HindsightExperienceReward(BaseReward):
    """
    Reward component that implements Hindsight Experience Replay (HER) style reward relabeling.
    Re-evaluates past experiences with achieved goals as if they were intended.
    """
    
    def __init__(self,
                 goal_key: str = "goal",
                 achieved_key: str = "achieved_goal",
                 base_reward_key: str = "base_reward",
                 hindsight_probability: float = 0.3,
                 relabeling_strategy: str = "final",
                 buffer_size: int = 1000,
                 **kwargs):
        """
        Initialize hindsight experience reward component.
        
        Args:
            goal_key: Key in context containing desired goal
            achieved_key: Key in context containing achieved goal
            base_reward_key: Key in context containing base reward
            hindsight_probability: Probability of applying hindsight relabeling
            relabeling_strategy: Strategy for relabeling ("final", "future", "episode")
            buffer_size: Size of experience buffer
        """
        super().__init__(**kwargs)
        self.goal_key = goal_key
        self.achieved_key = achieved_key
        self.base_reward_key = base_reward_key
        self.hindsight_probability = hindsight_probability
        self.relabeling_strategy = relabeling_strategy
        self.buffer_size = buffer_size
        
        # Experience buffer
        self.experience_buffer: Deque = deque(maxlen=buffer_size)
        self.current_episode_experiences = []
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate hindsight experience reward."""
        # Extract base reward
        base_reward = context.get(self.base_reward_key, 0.0)
        if not isinstance(base_reward, (int, float)):
            base_reward = 0.0
            
        # Store experience
        self._store_experience(context, base_reward)
        
        # Apply hindsight relabeling with some probability
        if np.random.random() < self.hindsight_probability:
            hindsight_reward = self._apply_hindsight_relabeling(context)
            if hindsight_reward is not None:
                return hindsight_reward
                
        return base_reward
        
    def _store_experience(self, context: Dict[str, Any], reward: float) -> None:
        """Store experience in buffer."""
        experience = {
            'context': context.copy(),
            'reward': reward,
            'timestamp': time.time()
        }
        
        # Add to current episode
        self.current_episode_experiences.append(experience)
        
        # Check if episode is done
        if context.get('done', False):
            # Move episode to main buffer
            self.experience_buffer.extend(self.current_episode_experiences)
            self.current_episode_experiences = []
            
    def _apply_hindsight_relabeling(self, context: Dict[str, Any]) -> Optional[float]:
        """Apply hindsight experience relabeling."""
        if not self.experience_buffer:
            return None
            
        # Get current achieved goal
        achieved_goal = context.get(self.achieved_key)
        if achieved_goal is None:
            return None
            
        # Select relabeling strategy
        if self.relabeling_strategy == "final":
            return self._final_goal_relabeling(achieved_goal)
        elif self.relabeling_strategy == "future":
            return self._future_goal_relabeling(context)
        elif self.relabeling_strategy == "episode":
            return self._episode_goal_relabeling(context)
        else:
            return None
            
    def _final_goal_relabeling(self, achieved_goal: Any) -> float:
        """Relabel with final achieved goal."""
        # Sample a random past experience
        if not self.experience_buffer:
            return 0.0
            
        past_experience = np.random.choice(list(self.experience_buffer))
        
        # Compute reward as if achieved_goal was the intended goal
        distance = self._compute_goal_distance(
            past_experience['context'].get(self.achieved_key),
            achieved_goal
        )
        
        # Reward is higher for closer goals
        hindsight_reward = 1.0 / (1.0 + distance)
        
        return hindsight_reward
        
    def _future_goal_relabeling(self, context: Dict[str, Any]) -> float:
        """Relabel with future achieved goals."""
        # This is a simplified version - in practice, you'd use actual future experiences
        current_achieved = context.get(self.achieved_key)
        
        if current_achieved is None:
            return 0.0
            
        # Sample from recent experiences as "future" goals
        recent_experiences = list(self.experience_buffer)[-10:]
        if not recent_experiences:
            return 0.0
            
        future_experience = np.random.choice(recent_experiences)
        future_goal = future_experience['context'].get(self.achieved_key)
        
        if future_goal is None:
            return 0.0
            
        # Compute reward for reaching this "future" goal
        distance = self._compute_goal_distance(current_achieved, future_goal)
        hindsight_reward = 1.0 / (1.0 + distance)
        
        return hindsight_reward
        
    def _episode_goal_relabeling(self, context: Dict[str, Any]) -> float:
        """Relabel with goals from the same episode."""
        if not self.current_episode_experiences:
            return 0.0
            
        # Sample a goal from current episode
        episode_experience = np.random.choice(self.current_episode_experiences)
        episode_goal = episode_experience['context'].get(self.achieved_key)
        
        current_achieved = context.get(self.achieved_key)
        
        if episode_goal is None or current_achieved is None:
            return 0.0
            
        # Compute reward for this episode goal
        distance = self._compute_goal_distance(current_achieved, episode_goal)
        hindsight_reward = 1.0 / (1.0 + distance)
        
        return hindsight_reward
        
    def _compute_goal_distance(self, achieved: Any, goal: Any) -> float:
        """Compute distance between achieved and goal states."""
        if achieved is None or goal is None:
            return float('inf')
            
        try:
            # Convert to numpy arrays
            achieved_array = np.array(achieved)
            goal_array = np.array(goal)
            
            # Compute Euclidean distance
            distance = np.linalg.norm(achieved_array - goal_array)
            return distance
            
        except (ValueError, TypeError):
            # Fallback for non-numerical goals
            if achieved == goal:
                return 0.0
            else:
                return 1.0
                
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get experience buffer statistics."""
        return {
            'buffer_size': len(self.experience_buffer),
            'current_episode_length': len(self.current_episode_experiences),
            'total_experiences': len(self.experience_buffer) + len(self.current_episode_experiences)
        }
