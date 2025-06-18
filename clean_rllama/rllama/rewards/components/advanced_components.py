# rllama/rewards/components/advanced_components.py

from typing import Dict, List, Any, Optional, Union
import numpy as np
import torch
from .base import BaseReward
import re
from collections import deque
import time

class ExplorationReward(BaseReward):
    """
    Rewards agent for exploring new states/actions.
    Uses count-based exploration bonuses.
    """
    
    def __init__(self, 
                 decay_factor: float = 0.99,
                 visit_threshold: int = 3,
                 reward_scale: float = 0.1,
                 novelty_window: int = 1000):
        """
        Args:
            decay_factor: Rate at which novelty reward decays with visits.
            visit_threshold: After this many visits, state is no longer novel.
            reward_scale: Scaling factor for the exploration reward.
            novelty_window: How many recent states to track for novelty.
        """
        super().__init__()
        self.decay_factor = decay_factor
        self.visit_threshold = visit_threshold
        self.reward_scale = reward_scale
        self.state_visit_counts = {}
        self.recent_states = deque(maxlen=novelty_window)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        # Get state representation
        state_repr = context.get("state_hash") or context.get("state_repr")
        
        if state_repr is None:
            # Try to extract state embedding
            state_embedding = context.get("state_embedding")
            if state_embedding is not None:
                if isinstance(state_embedding, torch.Tensor):
                    state_embedding = state_embedding.cpu().numpy()
                # Convert embedding to hashable string representation
                state_repr = hash(state_embedding.tobytes())
            else:
                # Last resort - try to get state
                state = context.get("state") or context.get("observation")
                if state is None:
                    return 0.0
                
                try:
                    if isinstance(state, torch.Tensor):
                        state = state.cpu().numpy()
                    state_repr = hash(str(state))
                except:
                    # If we can't hash it, we can't count visits
                    return 0.0
        
        # Convert to hashable if needed
        if not isinstance(state_repr, (str, int)):
            state_repr = str(state_repr)
            
        # Check if state is novel
        visit_count = self.state_visit_counts.get(state_repr, 0)
        
        # Update visit count
        self.state_visit_counts[state_repr] = visit_count + 1
        self.recent_states.append(state_repr)
        
        # Calculate novelty bonus
        if visit_count < self.visit_threshold:
            novelty_bonus = self.reward_scale * (self.decay_factor ** visit_count)
            return novelty_bonus
        else:
            return 0.0
    
    def reset(self):
        """Reset the exploration stats"""
        self.state_visit_counts = {}
        self.recent_states.clear()


class CurriculumReward(BaseReward):
    """
    Implements a curriculum learning approach by dynamically 
    adjusting reward thresholds and scales.
    """
    
    def __init__(self, 
                difficulty_thresholds: List[int], 
                base_rewards: List[float],
                difficulty_key: str = "difficulty", 
                progress_key: str = "step"):
        """
        Args:
            difficulty_thresholds: List of step thresholds for increasing difficulty.
            base_rewards: Corresponding base reward values for each difficulty level.
            difficulty_key: Key in context for difficulty level.
            progress_key: Key in context for tracking progress (e.g., step count).
        """
        super().__init__()
        if len(difficulty_thresholds) != len(base_rewards) - 1:
            raise ValueError("Must provide one more base reward than thresholds")
            
        self.difficulty_thresholds = difficulty_thresholds
        self.base_rewards = base_rewards
        self.difficulty_key = difficulty_key
        self.progress_key = progress_key
        self.current_level = 0
    
    def calculate(self, context: Dict[str, Any]) -> float:
        # Get current progress (defaults to 0)
        progress = context.get(self.progress_key, 0)
        
        # Check if we need to adjust difficulty
        for i, threshold in enumerate(self.difficulty_thresholds):
            if progress >= threshold:
                self.current_level = i + 1
        
        # Calculate reward based on current level
        difficulty_factor = context.get(self.difficulty_key, 1.0)
        base_reward = self.base_rewards[self.current_level]
        
        return base_reward * difficulty_factor
    
    def reset(self):
        """Reset to initial difficulty level"""
        self.current_level = 0


class CompositeReward(BaseReward):
    """
    A reward component that combines multiple other components.
    Useful for creating hierarchical reward structures.
    """
    
    def __init__(self, components: List[BaseReward], weights: Optional[List[float]] = None):
        """
        Args:
            components: List of reward components to combine.
            weights: Weight for each component (defaults to equal weights).
        """
        super().__init__()
        self.components = components
        
        if weights is None:
            # Equal weights by default
            self.weights = [1.0 / len(components)] * len(components)
        else:
            if len(weights) != len(components):
                raise ValueError("Number of weights must match number of components")
            # Normalize weights to sum to 1.0
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def calculate(self, context: Dict[str, Any]) -> float:
        total_reward = 0.0
        for component, weight in zip(self.components, self.weights):
            component_reward = component.calculate(context)
            total_reward += weight * component_reward
        
        return total_reward
    
    def reset(self):
        """Reset all sub-components"""
        for component in self.components:
            if hasattr(component, 'reset'):
                component.reset()


class SparseReward(BaseReward):
    """
    Converts dense rewards into sparse rewards by only giving
    rewards when certain conditions are met.
    """
    
    def __init__(self, 
                condition_key: str,
                condition_threshold: float,
                success_reward: float = 1.0,
                failure_reward: float = -0.1,
                baseline_reward: float = 0.0,
                terminal_only: bool = False):
        """
        Args:
            condition_key: Key in context to check condition against.
            condition_threshold: Threshold value for the condition.
            success_reward: Reward when condition is met.
            failure_reward: Reward when condition is explicitly failed.
            baseline_reward: Default reward when neither success nor failure.
            terminal_only: Only give rewards on terminal states.
        """
        super().__init__()
        self.condition_key = condition_key
        self.condition_threshold = condition_threshold
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.baseline_reward = baseline_reward
        self.terminal_only = terminal_only
    
    def calculate(self, context: Dict[str, Any]) -> float:
        # Check if this is a terminal state
        done = context.get("done", False)
        
        # If terminal_only is set, only continue if we're at a terminal state
        if self.terminal_only and not done:
            return self.baseline_reward
        
        # Check condition value in context
        condition_value = context.get(self.condition_key)
        
        # If condition key isn't in context, return baseline
        if condition_value is None:
            return self.baseline_reward
        
        # Compare against threshold
        if condition_value >= self.condition_threshold:
            return self.success_reward
        elif done:  # If terminal state and condition not met, it's a failure
            return self.failure_reward
        else:
            return self.baseline_reward


# rllama/rewards/components/advanced_components.py
# (continuation from previous code)

class LearningProgressReward(BaseReward):
    """
    Rewards based on improvement in performance over time.
    Useful for encouraging continuous learning.
    """
    
    def __init__(self, 
                 performance_key: str,
                 window_size: int = 100,
                 reward_scale: float = 0.1,
                 baseline_reward: float = 0.0):
        """
        Args:
            performance_key: Key in context for performance metric.
            window_size: How many past performance values to track.
            reward_scale: Scale factor for the learning progress reward.
            baseline_reward: Default reward when no progress can be calculated.
        """
        super().__init__()
        self.performance_key = performance_key
        self.window_size = window_size
        self.reward_scale = reward_scale
        self.baseline_reward = baseline_reward
        self.performance_history = deque(maxlen=window_size)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        # Get current performance metric
        current_performance = context.get(self.performance_key)
        
        if current_performance is None:
            return self.baseline_reward
        
        # Store current performance
        self.performance_history.append(current_performance)
        
        # Need at least 2 data points to calculate progress
        if len(self.performance_history) < 2:
            return self.baseline_reward
        
        # Calculate recent average performance (last 1/3 of window)
        recent_window = max(1, len(self.performance_history) // 3)
        recent_avg = sum(list(self.performance_history)[-recent_window:]) / recent_window
        
        # Calculate previous average performance
        prev_window = max(1, len(self.performance_history) - recent_window)
        prev_avg = sum(list(self.performance_history)[:prev_window]) / prev_window
        
        # Calculate improvement
        improvement = recent_avg - prev_avg
        
        # Scale the improvement to get the reward
        return self.reward_scale * improvement + self.baseline_reward
    
    def reset(self):
        """Reset learning history"""
        self.performance_history.clear()


class DiversityReward(BaseReward):
    """
    Rewards diversity in actions or outputs to encourage exploration.
    """
    
    def __init__(self,
                 target_key: str,
                 history_size: int = 100,
                 reward_scale: float = 0.1,
                 similarity_threshold: float = 0.8,
                 use_semantic: bool = False,
                 tokenizer=None):
        """
        Args:
            target_key: Key in context for the value to check for diversity.
            history_size: How many past values to track.
            reward_scale: Scale factor for the diversity reward.
            similarity_threshold: Threshold below which items are considered diverse.
            use_semantic: Whether to use semantic similarity (requires tokenizer).
            tokenizer: Tokenizer for semantic similarity (required if use_semantic=True).
        """
        super().__init__()
        self.target_key = target_key
        self.history = deque(maxlen=history_size)
        self.reward_scale = reward_scale
        self.similarity_threshold = similarity_threshold
        self.use_semantic = use_semantic
        
        if use_semantic and tokenizer is None:
            raise ValueError("Tokenizer required for semantic similarity")
        
        self.tokenizer = tokenizer
        self.embeddings = []
        
    def calculate(self, context: Dict[str, Any]) -> float:
        # Get current value
        current_value = context.get(self.target_key)
        
        if current_value is None:
            return 0.0
        
        # Empty history means this is new by definition
        if not self.history:
            self.history.append(current_value)
            if self.use_semantic:
                self.embeddings.append(self._get_embedding(current_value))
            return self.reward_scale  # Maximum diversity reward
        
        # Calculate similarity to history
        if self.use_semantic and self.tokenizer:
            # Get embedding for current value
            current_embedding = self._get_embedding(current_value)
            
            # Calculate semantic similarity to history
            similarities = [self._cosine_similarity(current_embedding, emb) for emb in self.embeddings]
            max_similarity = max(similarities)
        else:
            # Use string similarity for non-semantic comparison
            if isinstance(current_value, str):
                similarities = [self._string_similarity(current_value, val) for val in self.history
                              if isinstance(val, str)]
                max_similarity = max(similarities) if similarities else 0.0
            else:
                # For non-string types, check for exact matches
                max_similarity = 1.0 if current_value in self.history else 0.0
        
        # Calculate diversity (inverse of similarity)
        diversity = 1.0 - max_similarity
        
        # Add to history
        self.history.append(current_value)
        if self.use_semantic:
            self.embeddings.append(self._get_embedding(current_value))
        
        # Reward is scaled by diversity
        if diversity > self.similarity_threshold:
            return self.reward_scale * diversity
        else:
            return 0.0
    
    def _get_embedding(self, text):
        """Get embedding for text using tokenizer"""
        import torch
        
        if not isinstance(text, str):
            text = str(text)
            
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.tokenizer.model(**inputs)
            
            # Get the embeddings from the last hidden layer
            if hasattr(outputs, 'last_hidden_state'):
                # Get mean of token embeddings
                embedding = torch.mean(outputs.last_hidden_state, dim=1)
            else:
                embedding = outputs[0].mean(dim=1)
                
            return embedding.cpu().numpy().flatten()
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _string_similarity(self, str1, str2):
        """Calculate simple string similarity"""
        # Convert to lowercase and tokenize by spaces
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def reset(self):
        """Reset diversity history"""
        self.history.clear()
        self.embeddings = []


class SampleEfficiencyReward(BaseReward):
    """
    Rewards using fewer steps to accomplish goals.
    Encourages sample efficiency in RL.
    """
    
    def __init__(self,
                 success_key: str,
                 max_steps: int = 1000,
                 base_reward: float = 1.0,
                 step_penalty: float = 0.001):
        """
        Args:
            success_key: Key in context indicating goal achievement.
            max_steps: Maximum number of steps expected for task.
            base_reward: Base reward value for goal achievement.
            step_penalty: Penalty per step used.
        """
        super().__init__()
        self.success_key = success_key
        self.max_steps = max_steps
        self.base_reward = base_reward
        self.step_penalty = step_penalty
        self.step_count = 0
        self.goal_achieved = False
        
    def calculate(self, context: Dict[str, Any]) -> float:
        # Check if this is a reset signal
        reset = context.get("reset", False)
        if reset:
            self.reset()
            return 0.0
        
        # Increment step counter
        self.step_count += 1
        
        # Check if goal is achieved
        success = context.get(self.success_key, False)
        done = context.get("done", False)
        
        if success and not self.goal_achieved:
            # First time achieving goal
            self.goal_achieved = True
            
            # Calculate reward based on steps taken
            efficiency_factor = 1.0 - min(1.0, self.step_count / self.max_steps)
            efficiency_bonus = self.base_reward * efficiency_factor
            
            return self.base_reward + efficiency_bonus
        elif done and not self.goal_achieved:
            # Failed to achieve goal, apply step penalty
            return -self.step_penalty * self.step_count
        else:
            # Small penalty per step to encourage efficiency
            return -self.step_penalty
    
    def reset(self):
        """Reset counter for new episode"""
        self.step_count = 0
        self.goal_achieved = False
       
