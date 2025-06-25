#!/usr/bin/env python3
"""
Advanced reward components for sophisticated reward engineering scenarios.
These components implement cutting-edge techniques for complex reward shaping.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from collections import deque, defaultdict
import time
import warnings

from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class TemporalConsistencyReward(BaseReward):
    """
    Reward component that enforces temporal consistency across sequences.
    Penalizes sudden changes and rewards smooth transitions over time.
    """
    
    def __init__(self,
                 value_key: str = "value",
                 consistency_window: int = 5,
                 consistency_weight: float = 1.0,
                 change_threshold: float = 0.1,
                 smoothness_factor: float = 0.5,
                 **kwargs):
        """
        Initialize temporal consistency reward component.
        
        Args:
            value_key: Key in context containing value to check for consistency
            consistency_window: Number of previous values to consider
            consistency_weight: Weight for consistency reward
            change_threshold: Threshold for considering changes significant
            smoothness_factor: Factor for smoothness calculation
        """
        super().__init__(**kwargs)
        self.value_key = value_key
        self.consistency_window = consistency_window
        self.consistency_weight = consistency_weight
        self.change_threshold = change_threshold
        self.smoothness_factor = smoothness_factor
        
        # Track value history
        self.value_history: deque = deque(maxlen=consistency_window)
        self.change_history: deque = deque(maxlen=consistency_window)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate temporal consistency reward."""
        # Extract current value
        current_value = context.get(self.value_key)
        if current_value is None:
            return 0.0
            
        # Convert to float if possible
        try:
            if isinstance(current_value, (list, tuple)):
                current_value = np.array(current_value)
                value_magnitude = np.linalg.norm(current_value)
            else:
                value_magnitude = float(current_value)
        except (ValueError, TypeError):
            return 0.0
            
        # Add to history
        self.value_history.append(value_magnitude)
        
        # Need at least 2 values for consistency check
        if len(self.value_history) < 2:
            return 0.0
            
        # Calculate change from previous value
        previous_value = self.value_history[-2]
        change = abs(value_magnitude - previous_value)
        self.change_history.append(change)
        
        # Calculate consistency metrics
        consistency_reward = self._calculate_consistency_reward()
        smoothness_reward = self._calculate_smoothness_reward()
        
        # Combine rewards
        total_reward = (
            self.consistency_weight * consistency_reward +
            self.smoothness_factor * smoothness_reward
        )
        
        return total_reward
        
    def _calculate_consistency_reward(self) -> float:
        """Calculate reward based on value consistency."""
        if len(self.value_history) < 2:
            return 0.0
            
        # Calculate variance of recent values
        recent_values = list(self.value_history)
        variance = np.var(recent_values)
        
        # Lower variance = higher consistency
        consistency_score = 1.0 / (1.0 + variance)
        
        # Penalty for large changes
        recent_change = self.change_history[-1] if self.change_history else 0.0
        if recent_change > self.change_threshold:
            change_penalty = (recent_change - self.change_threshold) ** 2
            consistency_score -= change_penalty
            
        return max(0.0, consistency_score)
        
    def _calculate_smoothness_reward(self) -> float:
        """Calculate reward based on smoothness of changes."""
        if len(self.change_history) < 2:
            return 0.0
            
        # Calculate second derivative (acceleration)
        changes = list(self.change_history)
        accelerations = []
        
        for i in range(1, len(changes)):
            acceleration = abs(changes[i] - changes[i-1])
            accelerations.append(acceleration)
            
        if not accelerations:
            return 0.0
            
        # Lower acceleration = smoother changes
        avg_acceleration = np.mean(accelerations)
        smoothness_score = 1.0 / (1.0 + avg_acceleration)
        
        return smoothness_score
        
    def get_consistency_stats(self) -> Dict[str, float]:
        """Get current consistency statistics."""
        if not self.value_history:
            return {"variance": 0.0, "avg_change": 0.0, "smoothness": 0.0}
            
        variance = np.var(list(self.value_history))
        avg_change = np.mean(list(self.change_history)) if self.change_history else 0.0
        
        # Calculate smoothness
        if len(self.change_history) >= 2:
            changes = list(self.change_history)
            accelerations = [abs(changes[i] - changes[i-1]) for i in range(1, len(changes))]
            smoothness = 1.0 / (1.0 + np.mean(accelerations))
        else:
            smoothness = 1.0
            
        return {
            "variance": variance,
            "avg_change": avg_change,
            "smoothness": smoothness,
            "history_length": len(self.value_history)
        }

@register_reward_component
class MultiObjectiveReward(BaseReward):
    """
    Reward component that handles multiple competing objectives.
    Uses Pareto optimization and scalarization techniques.
    """
    
    def __init__(self,
                 objective_keys: List[str],
                 objective_weights: Optional[List[float]] = None,
                 scalarization_method: str = "weighted_sum",
                 pareto_front_size: int = 100,
                 reference_point: Optional[List[float]] = None,
                 **kwargs):
        """
        Initialize multi-objective reward component.
        
        Args:
            objective_keys: List of keys in context containing objective values
            objective_weights: Weights for each objective (if None, equal weights)
            scalarization_method: Method to combine objectives ("weighted_sum", "tchebycheff", "hypervolume")
            pareto_front_size: Size of Pareto front to maintain
            reference_point: Reference point for hypervolume calculation
        """
        super().__init__(**kwargs)
        self.objective_keys = objective_keys
        self.num_objectives = len(objective_keys)
        
        # Set default weights if not provided
        if objective_weights is None:
            self.objective_weights = [1.0 / self.num_objectives] * self.num_objectives
        else:
            if len(objective_weights) != self.num_objectives:
                raise ValueError("Number of weights must match number of objectives")
            self.objective_weights = objective_weights
            
        self.scalarization_method = scalarization_method
        self.pareto_front_size = pareto_front_size
        self.reference_point = reference_point or [0.0] * self.num_objectives
        
        # Track Pareto front
        self.pareto_front: List[Tuple[List[float], float]] = []
        self.objective_history: List[List[float]] = []
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate multi-objective reward."""
        # Extract objective values
        objectives = []
        for key in self.objective_keys:
            value = context.get(key, 0.0)
            try:
                objectives.append(float(value))
            except (ValueError, TypeError):
                objectives.append(0.0)
                
        # Add to history
        self.objective_history.append(objectives)
        
        # Update Pareto front
        self._update_pareto_front(objectives)
        
        # Calculate scalarized reward
        if self.scalarization_method == "weighted_sum":
            reward = self._weighted_sum_scalarization(objectives)
        elif self.scalarization_method == "tchebycheff":
            reward = self._tchebycheff_scalarization(objectives)
        elif self.scalarization_method == "hypervolume":
            reward = self._hypervolume_scalarization(objectives)
        else:
            reward = self._weighted_sum_scalarization(objectives)
            
        return reward
        
    def _weighted_sum_scalarization(self, objectives: List[float]) -> float:
        """Weighted sum scalarization."""
        return sum(w * obj for w, obj in zip(self.objective_weights, objectives))
        
    def _tchebycheff_scalarization(self, objectives: List[float]) -> float:
        """Tchebycheff scalarization."""
        # Find ideal point (maximum values seen so far)
        if not self.objective_history:
            ideal_point = objectives
        else:
            ideal_point = [max(obj_vals) for obj_vals in zip(*self.objective_history)]
            
        # Calculate Tchebycheff distance
        weighted_diffs = [w * abs(ideal - obj) 
                         for w, ideal, obj in zip(self.objective_weights, ideal_point, objectives)]
        
        # Return negative of maximum weighted difference (we want to minimize this)
        return -max(weighted_diffs)
        
    def _hypervolume_scalarization(self, objectives: List[float]) -> float:
        """Hypervolume-based scalarization."""
        # Simplified hypervolume contribution calculation
        # In practice, you'd use a proper hypervolume algorithm
        
        # Calculate dominated hypervolume
        dominated_volume = 1.0
        for i, (obj_val, ref_val) in enumerate(zip(objectives, self.reference_point)):
            if obj_val > ref_val:
                dominated_volume *= (obj_val - ref_val)
            else:
                dominated_volume = 0.0
                break
                
        return dominated_volume
        
    def _update_pareto_front(self, objectives: List[float]) -> None:
        """Update the Pareto front with new objectives."""
        # Check if new point dominates any existing points
        new_front = []
        dominated = False
        
        for front_objectives, _ in self.pareto_front:
            if self._dominates(objectives, front_objectives):
                # New point dominates this point, don't add it to new front
                continue
            elif self._dominates(front_objectives, objectives):
                # Existing point dominates new point
                dominated = True
                new_front.append((front_objectives, 0.0))  # Placeholder reward
            else:
                # Neither dominates, keep both
                new_front.append((front_objectives, 0.0))
                
        # Add new point if not dominated
        if not dominated:
            new_front.append((objectives, 0.0))
            
        # Limit front size
        if len(new_front) > self.pareto_front_size:
            # Keep most diverse points (simplified diversity preservation)
            new_front = self._preserve_diversity(new_front)
            
        self.pareto_front = new_front
        
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)."""
        at_least_one_better = False
        
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False
            elif v1 > v2:
                at_least_one_better = True
                
        return at_least_one_better
        
    def _preserve_diversity(self, front: List[Tuple[List[float], float]]) -> List[Tuple[List[float], float]]:
        """Preserve diversity in Pareto front (simplified)."""
        if len(front) <= self.pareto_front_size:
            return front
            
        # Simple diversity preservation: keep points with maximum distance
        selected = [front[0]]  # Always keep first point
        
        for _ in range(self.pareto_front_size - 1):
            max_min_distance = -1
            best_candidate = None
            
            for candidate_obj, _ in front:
                if any(np.array_equal(candidate_obj, sel_obj) for sel_obj, _ in selected):
                    continue
                    
                # Calculate minimum distance to selected points
                min_distance = float('inf')
                for selected_obj, _ in selected:
                    distance = np.linalg.norm(np.array(candidate_obj) - np.array(selected_obj))
                    min_distance = min(min_distance, distance)
                    
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = (candidate_obj, 0.0)
                    
            if best_candidate:
                selected.append(best_candidate)
                
        return selected
        
    def get_pareto_statistics(self) -> Dict[str, Any]:
        """Get Pareto front statistics."""
        if not self.pareto_front:
            return {"front_size": 0, "hypervolume": 0.0}
            
        front_objectives = [obj for obj, _ in self.pareto_front]
        
        # Calculate hypervolume (simplified)
        total_hypervolume = 0.0
        for objectives in front_objectives:
            volume = 1.0
            for obj_val, ref_val in zip(objectives, self.reference_point):
                if obj_val > ref_val:
                    volume *= (obj_val - ref_val)
                else:
                    volume = 0.0
                    break
            total_hypervolume += volume
            
        return {
            "front_size": len(self.pareto_front),
            "hypervolume": total_hypervolume,
            "objective_ranges": self._calculate_objective_ranges(front_objectives)
        }
        
    def _calculate_objective_ranges(self, front_objectives: List[List[float]]) -> Dict[str, Tuple[float, float]]:
        """Calculate ranges for each objective in the Pareto front."""
        if not front_objectives:
            return {}
            
        ranges = {}
        for i, key in enumerate(self.objective_keys):
            values = [obj[i] for obj in front_objectives]
            ranges[key] = (min(values), max(values))
            
        return ranges

@register_reward_component
class HierarchicalReward(BaseReward):
    """
    Reward component that implements hierarchical reward structures.
    Supports multiple levels of abstraction and goal decomposition.
    """
    
    def __init__(self,
                 hierarchy_levels: List[str],
                 level_weights: Optional[List[float]] = None,
                 goal_key: str = "goal",
                 state_key: str = "state",
                 completion_thresholds: Optional[List[float]] = None,
                 **kwargs):
        """
        Initialize hierarchical reward component.
        
        Args:
            hierarchy_levels: List of hierarchy level names (low to high)
            level_weights: Weights for each hierarchy level
            goal_key: Key in context containing hierarchical goals
            state_key: Key in context containing current state
            completion_thresholds: Thresholds for considering each level complete
        """
        super().__init__(**kwargs)
        self.hierarchy_levels = hierarchy_levels
        self.num_levels = len(hierarchy_levels)
        
        # Set default weights if not provided
        if level_weights is None:
            # Higher levels get exponentially higher weights
            self.level_weights = [2.0 ** i for i in range(self.num_levels)]
        else:
            if len(level_weights) != self.num_levels:
                raise ValueError("Number of weights must match number of hierarchy levels")
            self.level_weights = level_weights
            
        self.goal_key = goal_key
        self.state_key = state_key
        
        # Set default completion thresholds
        if completion_thresholds is None:
            self.completion_thresholds = [0.1] * self.num_levels
        else:
            if len(completion_thresholds) != self.num_levels:
                raise ValueError("Number of thresholds must match number of hierarchy levels")
            self.completion_thresholds = completion_thresholds
            
        # Track completion status for each level
        self.level_completion: Dict[str, bool] = {level: False for level in hierarchy_levels}
        self.level_progress: Dict[str, float] = {level: 0.0 for level in hierarchy_levels}
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate hierarchical reward."""
        # Extract goals and state
        goals = context.get(self.goal_key, {})
        state = context.get(self.state_key)
        
        if not goals or state is None:
            return 0.0
            
        total_reward = 0.0
        
        # Calculate reward for each hierarchy level
        for i, level in enumerate(self.hierarchy_levels):
            level_goal = goals.get(level)
            if level_goal is None:
                continue
                
            # Calculate progress toward this level's goal
            progress = self._calculate_level_progress(state, level_goal, level)
            self.level_progress[level] = progress
            
            # Check completion
            completed = progress >= (1.0 - self.completion_thresholds[i])
            self.level_completion[level] = completed
            
            # Calculate level reward
            level_reward = self._calculate_level_reward(progress, completed, i)
            
            # Add weighted contribution
            total_reward += self.level_weights[i] * level_reward
            
        # Apply hierarchical bonus
        hierarchical_bonus = self._calculate_hierarchical_bonus()
        total_reward += hierarchical_bonus
        
        return total_reward
        
    def _calculate_level_progress(self, state: Any, goal: Any, level: str) -> float:
        """Calculate progress toward a specific level's goal."""
        try:
            # Convert to numpy arrays if possible
            state_array = np.array(state)
            goal_array = np.array(goal)
            
            # Calculate distance-based progress
            distance = np.linalg.norm(state_array - goal_array)
            
            # Convert distance to progress (0 = far, 1 = at goal)
            # Use exponential decay for smoother progress
            progress = math.exp(-distance)
            
            return min(1.0, max(0.0, progress))
            
        except (ValueError, TypeError):
            # Fallback for non-numerical goals
            if state == goal:
                return 1.0
            else:
                return 0.0
                
    def _calculate_level_reward(self, progress: float, completed: bool, level_index: int) -> float:
        """Calculate reward for a specific hierarchy level."""
        # Base reward from progress
        base_reward = progress
        
        # Completion bonus
        if completed:
            completion_bonus = 1.0
        else:
            completion_bonus = 0.0
            
        # Level-specific scaling
        level_scaling = 1.0 + 0.1 * level_index  # Higher levels get slight bonus
        
        return (base_reward + completion_bonus) * level_scaling
        
    def _calculate_hierarchical_bonus(self) -> float:
        """Calculate bonus for hierarchical completion patterns."""
        # Bonus for completing levels in order (bottom-up)
        sequential_bonus = 0.0
        
        for i, level in enumerate(self.hierarchy_levels):
            if self.level_completion[level]:
                # Check if all lower levels are also complete
                lower_levels_complete = all(
                    self.level_completion[self.hierarchy_levels[j]] 
                    for j in range(i)
                )
                
                if lower_levels_complete:
                    sequential_bonus += 0.1 * (i + 1)  # Higher levels give more bonus
                    
        # Bonus for overall completion rate
        completion_rate = sum(self.level_completion.values()) / self.num_levels
        completion_bonus = completion_rate ** 2  # Quadratic bonus
        
        return sequential_bonus + completion_bonus
        
    def get_hierarchy_status(self) -> Dict[str, Any]:
        """Get current hierarchy status."""
        return {
            "level_completion": self.level_completion.copy(),
            "level_progress": self.level_progress.copy(),
            "overall_completion": sum(self.level_completion.values()) / self.num_levels,
            "completed_levels": [level for level, completed in self.level_completion.items() if completed]
        }

@register_reward_component
class ContrastiveReward(BaseReward):
    """
    Reward component that uses contrastive learning principles.
    Encourages similarity to positive examples and dissimilarity to negative examples.
    """
    
    def __init__(self,
                 positive_examples_key: str = "positive_examples",
                 negative_examples_key: str = "negative_examples",
                 current_example_key: str = "current_example",
                 similarity_metric: str = "cosine",
                 temperature: float = 0.1,
                 margin: float = 0.5,
                 **kwargs):
        """
        Initialize contrastive reward component.
        
        Args:
            positive_examples_key: Key in context containing positive examples
            negative_examples_key: Key in context containing negative examples
            current_example_key: Key in context containing current example
            similarity_metric: Similarity metric to use ("cosine", "euclidean", "dot")
            temperature: Temperature parameter for contrastive loss
            margin: Margin for contrastive loss
        """
        super().__init__(**kwargs)
        self.positive_examples_key = positive_examples_key
        self.negative_examples_key = negative_examples_key
        self.current_example_key = current_example_key
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.margin = margin
        
        # Track example history for dynamic contrast
        self.positive_history: deque = deque(maxlen=100)
        self.negative_history: deque = deque(maxlen=100)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate contrastive reward."""
        # Extract examples
        current_example = context.get(self.current_example_key)
        positive_examples = context.get(self.positive_examples_key, [])
        negative_examples = context.get(self.negative_examples_key, [])
        
        if current_example is None:
            return 0.0
            
        # Convert to numpy array
        try:
            current_array = np.array(current_example)
        except (ValueError, TypeError):
            return 0.0
            
        # Add to history
        if positive_examples:
            self.positive_history.extend(positive_examples)
        if negative_examples:
            self.negative_history.extend(negative_examples)
            
        # Use both provided examples and history
        all_positives = list(positive_examples) + list(self.positive_history)
        all_negatives = list(negative_examples) + list(self.negative_history)
        
        # Calculate contrastive reward
        if self.similarity_metric == "cosine":
            reward = self._cosine_contrastive_reward(current_array, all_positives, all_negatives)
        elif self.similarity_metric == "euclidean":
            reward = self._euclidean_contrastive_reward(current_array, all_positives, all_negatives)
        elif self.similarity_metric == "dot":
            reward = self._dot_contrastive_reward(current_array, all_positives, all_negatives)
        else:
            reward = self._cosine_contrastive_reward(current_array, all_positives, all_negatives)
            
        return reward
        
    def _cosine_contrastive_reward(self, 
                                 current: np.ndarray, 
                                 positives: List[Any], 
                                 negatives: List[Any]) -> float:
        """Calculate contrastive reward using cosine similarity."""
        # Calculate similarities to positive examples
        positive_similarities = []
        for pos_example in positives:
            try:
                pos_array = np.array(pos_example)
                if pos_array.shape == current.shape:
                    similarity = self._cosine_similarity(current, pos_array)
                    positive_similarities.append(similarity)
            except (ValueError, TypeError):
                continue
                
        # Calculate similarities to negative examples
        negative_similarities = []
        for neg_example in negatives:
            try:
                neg_array = np.array(neg_example)
                if neg_array.shape == current.shape:
                    similarity = self._cosine_similarity(current, neg_array)
                    negative_similarities.append(similarity)
            except (ValueError, TypeError):
                continue
                
        # Calculate contrastive reward
        if not positive_similarities and not negative_similarities:
            return 0.0
            
        # Average positive similarity (should be high)
        avg_positive_sim = np.mean(positive_similarities) if positive_similarities else 0.0
        
        # Average negative similarity (should be low)
        avg_negative_sim = np.mean(negative_similarities) if negative_similarities else 0.0
        
        # Contrastive reward: positive similarity - negative similarity
        contrastive_reward = avg_positive_sim - avg_negative_sim
        
        # Apply temperature scaling
        contrastive_reward = contrastive_reward / self.temperature
        
        # Apply margin
        contrastive_reward = max(0.0, contrastive_reward - self.margin)
        
        return contrastive_reward
        
    def _euclidean_contrastive_reward(self, 
                                    current: np.ndarray, 
                                    positives: List[Any], 
                                    negatives: List[Any]) -> float:
        """Calculate contrastive reward using Euclidean distance."""
        # Calculate distances to positive examples (should be small)
        positive_distances = []
        for pos_example in positives:
            try:
                pos_array = np.array(pos_example)
                if pos_array.shape == current.shape:
                    distance = np.linalg.norm(current - pos_array)
                    positive_distances.append(distance)
            except (ValueError, TypeError):
                continue
                
        # Calculate distances to negative examples (should be large)
        negative_distances = []
        for neg_example in negatives:
            try:
                neg_array = np.array(neg_example)
                if neg_array.shape == current.shape:
                    distance = np.linalg.norm(current - neg_array)
                    negative_distances.append(distance)
            except (ValueError, TypeError):
                continue
                
        if not positive_distances and not negative_distances:
            return 0.0
            
        # Average distances
        avg_positive_dist = np.mean(positive_distances) if positive_distances else 0.0
        avg_negative_dist = np.mean(negative_distances) if negative_distances else 0.0
        
        # Contrastive reward: negative distance - positive distance
        # (we want small positive distances and large negative distances)
        contrastive_reward = avg_negative_dist - avg_positive_dist
        
        # Apply temperature scaling
        contrastive_reward = contrastive_reward / self.temperature
        
        return max(0.0, contrastive_reward)
        
    def _dot_contrastive_reward(self, 
                              current: np.ndarray, 
                              positives: List[Any], 
                              negatives: List[Any]) -> float:
        """Calculate contrastive reward using dot product."""
        # Calculate dot products with positive examples
        positive_dots = []
        for pos_example in positives:
            try:
                pos_array = np.array(pos_example)
                if pos_array.shape == current.shape:
                    dot_product = np.dot(current, pos_array)
                    positive_dots.append(dot_product)
            except (ValueError, TypeError):
                continue
                
        # Calculate dot products with negative examples
        negative_dots = []
        for neg_example in negatives:
            try:
                neg_array = np.array(neg_example)
                if neg_array.shape == current.shape:
                    dot_product = np.dot(current, neg_array)
                    negative_dots.append(dot_product)
            except (ValueError, TypeError):
                continue
                
        if not positive_dots and not negative_dots:
            return 0.0
            
        # Average dot products
        avg_positive_dot = np.mean(positive_dots) if positive_dots else 0.0
        avg_negative_dot = np.mean(negative_dots) if negative_dots else 0.0
        
        # Contrastive reward
        contrastive_reward = avg_positive_dot - avg_negative_dot
        
        # Apply temperature scaling
        contrastive_reward = contrastive_reward / self.temperature
        
        return contrastive_reward
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
        
    def get_contrastive_statistics(self) -> Dict[str, Any]:
        """Get contrastive learning statistics."""
        return {
            "positive_examples_count": len(self.positive_history),
            "negative_examples_count": len(self.negative_history),
            "temperature": self.temperature,
            "margin": self.margin,
            "similarity_metric": self.similarity_metric
        }
