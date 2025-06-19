

import numpy as np
from typing import Dict, Any, List, Optional
from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class DiversityReward(BaseReward):
    """Rewards diversity in agent actions or outputs to encourage exploration."""
    
    def __init__(self, history_size: int = 10, key: str = "action", strength: float = 0.5):
        """
        Initialize the diversity reward.
        
        Args:
            history_size: Number of past actions to consider
            key: Key in the context dictionary for the action/output
            strength: Scaling factor for the reward
        """
        super().__init__()
        self.history = []
        self.history_size = history_size
        self.key = key
        self.strength = strength
    
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate diversity reward based on action history."""
        action = context.get(self.key)
        
        if action is None:
            return 0.0
            
        # Convert action to hashable type if needed
        if isinstance(action, np.ndarray):
            action = tuple(action.flatten())
        
        # Calculate diversity by checking how often this action appears in history
        if not self.history:
            diversity = 1.0  # Maximum diversity for first action
        else:
            occurrences = sum(1 for a in self.history if a == action)
            if occurrences == 0:
                diversity = 1.0  # Maximum diversity for new actions
            else:
                # Lower diversity for repeated actions
                diversity = 1.0 / (occurrences + 1)
        
        # Update history
        self.history.append(action)
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
        return self.strength * diversity
        
    def reset(self) -> None:
        """Reset action history."""
        self.history = []


@register_reward_component
class CuriosityReward(BaseReward):
    """
    Implements intrinsic curiosity reward based on prediction error.
    Rewards the agent for visiting states that are hard to predict.
    """
    
    def __init__(self, scaling: float = 1.0, decay: float = 0.999):
        """
        Initialize the curiosity reward.
        
        Args:
            scaling: Scaling factor for the reward
            decay: Decay factor for the reward over time
        """
        super().__init__()
        self.scaling = scaling
        self.decay = decay
        self.step = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate the curiosity reward based on prediction error.
        
        The context should contain 'prediction_error' key with the
        error between predicted and actual next state.
        """
        prediction_error = context.get('prediction_error')
        
        if prediction_error is None:
            return 0.0
            
        # Calculate decayed scaling factor
        current_scaling = self.scaling * (self.decay ** self.step)
        self.step += 1
        
        # Return scaled prediction error as reward
        return current_scaling * prediction_error
        
    def reset(self) -> None:
        """Reset step counter."""
        self.step = 0


@register_reward_component
class ProgressReward(BaseReward):
    """
    Rewards the agent for making progress toward a goal.
    Uses distance metrics to measure progress.
    """
    
    def __init__(self, goal_key: str = 'goal', state_key: str = 'state', 
                 distance_fn: str = 'euclidean', scaling: float = 1.0):
        """
        Initialize the progress reward.
        
        Args:
            goal_key: Key in context for the goal state
            state_key: Key in context for the current state
            distance_fn: Distance function ('euclidean', 'manhattan', 'cosine')
            scaling: Scaling factor for the reward
        """
        super().__init__()
        self.goal_key = goal_key
        self.state_key = state_key
        self.distance_fn = distance_fn
        self.scaling = scaling
        self.last_distance = None
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate progress reward based on distance reduction to goal.
        """
        goal = context.get(self.goal_key)
        state = context.get(self.state_key)
        
        if goal is None or state is None:
            return 0.0
            
        # Convert to numpy arrays if they aren't already
        if not isinstance(goal, np.ndarray):
            goal = np.array(goal)
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        # Calculate distance to goal
        if self.distance_fn == 'euclidean':
            distance = np.linalg.norm(state - goal)
        elif self.distance_fn == 'manhattan':
            distance = np.sum(np.abs(state - goal))
        elif self.distance_fn == 'cosine':
            norm_state = np.linalg.norm(state)
            norm_goal = np.linalg.norm(goal)
            if norm_state > 0 and norm_goal > 0:
                distance = 1.0 - np.dot(state, goal) / (norm_state * norm_goal)
            else:
                distance = 1.0
        else:
            # Default to euclidean
            distance = np.linalg.norm(state - goal)
            
        # Calculate progress (reduction in distance)
        if self.last_distance is None:
            progress = 0.0  # No progress on first call
        else:
            progress = self.last_distance - distance
            
        # Update last distance
        self.last_distance = distance
        
        # Return scaled progress reward
        return self.scaling * progress
        
    def reset(self) -> None:
        """Reset last distance."""
        self.last_distance = None
