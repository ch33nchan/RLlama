

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from ...rewards.base import BaseReward
from ...rewards.registry import register_reward_component

@register_reward_component
class MetaLearningReward(BaseReward):
    """
    A meta-learning reward that adapts to the learning progress of the agent.
    Uses trajectory information to estimate learning progress.
    """
    
    def __init__(self, 
                window_size: int = 100,
                progress_scale: float = 1.0,
                smoothing: float = 0.1):
        """
        Initialize the meta-learning reward.
        
        Args:
            window_size: Size of history window for learning progress calculation
            progress_scale: Scaling factor for progress rewards
            smoothing: Smoothing factor for learning progress estimates
        """
        super().__init__()
        self.window_size = window_size
        self.progress_scale = progress_scale
        self.smoothing = smoothing
        
        self.performance_history = []
        self.avg_performance = 0.0
        self.prev_progress = 0.0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate reward based on learning progress.
        
        Args:
            context: Must contain 'performance' metric value
            
        Returns:
            Meta-learning reward based on performance progress
        """
        performance = context.get('performance')
        
        if performance is None:
            return 0.0
            
        # Add to history
        self.performance_history.append(performance)
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            
        # Calculate performance derivatives (learning progress)
        if len(self.performance_history) < 2:
            progress = 0.0
        else:
            # Slope of recent performance (change in performance)
            recent = self.performance_history[-min(10, len(self.performance_history)):]
            if len(recent) >= 2:
                # Simple linear regression slope
                x = np.arange(len(recent))
                progress = np.polyfit(x, recent, 1)[0]  # Slope of the fit
            else:
                progress = 0.0
        
        # Smooth progress estimate
        progress = self.smoothing * progress + (1 - self.smoothing) * self.prev_progress
        self.prev_progress = progress
        
        # Learning progress reward (positive when improving)
        reward = self.progress_scale * max(0, progress)
        
        # Add extra information to context
        context['learning_progress'] = progress
        
        return reward
        
    def reset(self) -> None:
        """Reset performance history."""
        self.performance_history = []
        self.avg_performance = 0.0
        self.prev_progress = 0.0


@register_reward_component
class HindsightExperienceReward(BaseReward):
    """
    Implements hindsight experience reward - rewarding the agent for
    achieving any goal, not just the originally intended one.
    Helps with sparse reward problems.
    """
    
    def __init__(self, 
                goal_fn: Optional[Callable] = None,
                achievement_bonus: float = 1.0,
                future_discount: float = 0.9):
        """
        Initialize the hindsight experience reward.
        
        Args:
            goal_fn: Function that checks if a state achieves a goal
            achievement_bonus: Bonus for achieving any goal
            future_discount: Discount factor for future achievements
        """
        super().__init__()
        self.goal_fn = goal_fn or self._default_goal_fn
        self.achievement_bonus = achievement_bonus
        self.future_discount = future_discount
        self.trajectory = []
        self.achieved_goals = set()
        
    def _default_goal_fn(self, state: Any, goal: Any) -> bool:
        """Default goal checking function."""
        if isinstance(state, np.ndarray) and isinstance(goal, np.ndarray):
            # Check if close enough to goal
            distance = np.linalg.norm(state - goal)
            return distance < 0.1
        return False
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate hindsight reward.
        
        Args:
            context: Must contain 'state', 'goal', and 'done' keys
            
        Returns:
            Hindsight reward
        """
        state = context.get('state')
        goal = context.get('goal')
        done = context.get('done', False)
        
        if state is None or goal is None:
            return 0.0
            
        # Add to trajectory
        self.trajectory.append(state)
        
        # Check if current state is a goal
        is_goal = self.goal_fn(state, goal)
        
        reward = 0.0
        
        # If this state achieves the original goal
        if is_goal:
            reward += self.achievement_bonus
            
        # Check if this state achieves any previous state as goal
        # (hindsight experience)
        for past_state in self.trajectory[:-1]:  # Exclude current state
            if self.goal_fn(state, past_state) and tuple(past_state) not in self.achieved_goals:
                # We achieved a new goal that wasn't achieved before
                reward += self.achievement_bonus * self.future_discount
                self.achieved_goals.add(tuple(past_state))
                
        # On episode end, reset trajectory
        if done:
            self.reset()
            
        return reward
        
    def reset(self) -> None:
        """Reset trajectory and achieved goals."""
        self.trajectory = []
        self.achieved_goals = set()
