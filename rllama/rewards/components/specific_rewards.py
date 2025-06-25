#!/usr/bin/env python3
"""
Specific reward components for common RL scenarios.
These components implement sophisticated reward functions for diversity, curiosity, progress tracking, etc.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Union, List, Deque
from collections import deque, defaultdict
import time

from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class DiversityReward(BaseReward):
    """
    Reward component that encourages diversity in actions or states.
    Tracks history and penalizes repetitive behavior.
    """
    
    def __init__(self, 
                 history_size: int = 10,
                 key: str = "action",
                 strength: float = 1.0,
                 diversity_metric: str = "entropy",
                 min_diversity_threshold: float = 0.1,
                 **kwargs):
        """
        Initialize diversity reward component.
        
        Args:
            history_size: Number of recent items to track for diversity
            key: Key in context to extract values for diversity calculation
            strength: Scaling factor for the reward
            diversity_metric: Method to calculate diversity ("entropy", "unique_count", "variance")
            min_diversity_threshold: Minimum diversity threshold for reward
        """
        super().__init__(**kwargs)
        self.history_size = history_size
        self.key = key
        self.strength = strength
        self.diversity_metric = diversity_metric
        self.min_diversity_threshold = min_diversity_threshold
        
        # Track history
        self.history: Deque = deque(maxlen=history_size)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate diversity-based reward."""
        # Extract value from context
        value = context.get(self.key)
        if value is None:
            return 0.0
            
        # Add to history
        self.history.append(value)
        
        # Need at least 2 items for diversity calculation
        if len(self.history) < 2:
            return 0.0
            
        # Calculate diversity based on metric
        if self.diversity_metric == "entropy":
            diversity = self._calculate_entropy()
        elif self.diversity_metric == "unique_count":
            diversity = self._calculate_unique_count()
        elif self.diversity_metric == "variance":
            diversity = self._calculate_variance()
        else:
            diversity = self._calculate_entropy()  # Default
            
        # Apply threshold and scaling
        if diversity < self.min_diversity_threshold:
            return -self.strength * (self.min_diversity_threshold - diversity)
        else:
            return self.strength * diversity
            
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of history."""
        if not self.history:
            return 0.0
            
        # Count frequencies
        counts = defaultdict(int)
        for item in self.history:
            counts[str(item)] += 1
            
        # Calculate probabilities
        total = len(self.history)
        entropy = 0.0
        
        for count in counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)
                
        # Normalize by maximum possible entropy
        max_entropy = math.log2(min(len(counts), total))
        return entropy / max_entropy if max_entropy > 0 else 0.0
        
    def _calculate_unique_count(self) -> float:
        """Calculate ratio of unique items to total items."""
        if not self.history:
            return 0.0
            
        unique_items = len(set(str(item) for item in self.history))
        return unique_items / len(self.history)
        
    def _calculate_variance(self) -> float:
        """Calculate variance of numerical values."""
        if not self.history:
            return 0.0
            
        # Convert to numbers if possible
        try:
            values = [float(item) for item in self.history]
            if len(values) < 2:
                return 0.0
            return np.var(values) / (np.var(values) + 1.0)  # Normalized variance
        except (ValueError, TypeError):
            # Fall back to unique count for non-numerical data
            return self._calculate_unique_count()

@register_reward_component
class CuriosityReward(BaseReward):
    """
    Reward component based on prediction error or novelty.
    Encourages exploration of uncertain or novel states.
    """
    
    def __init__(self,
                 scaling: float = 1.0,
                 decay: float = 0.99,
                 prediction_error_key: str = "prediction_error",
                 novelty_key: str = "novelty",
                 use_decay: bool = True,
                 **kwargs):
        """
        Initialize curiosity reward component.
        
        Args:
            scaling: Initial scaling factor for curiosity reward
            decay: Decay factor for scaling over time
            prediction_error_key: Key in context for prediction error
            novelty_key: Key in context for novelty score
            use_decay: Whether to apply decay to scaling factor
        """
        super().__init__(**kwargs)
        self.initial_scaling = scaling
        self.current_scaling = scaling
        self.decay = decay
        self.prediction_error_key = prediction_error_key
        self.novelty_key = novelty_key
        self.use_decay = use_decay
        self.step_count = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate curiosity-based reward."""
        # Get prediction error or novelty
        prediction_error = context.get(self.prediction_error_key, 0.0)
        novelty = context.get(self.novelty_key, 0.0)
        
        # Use prediction error if available, otherwise use novelty
        curiosity_signal = prediction_error if prediction_error > 0 else novelty
        
        # Apply current scaling
        reward = self.current_scaling * curiosity_signal
        
        # Apply decay if enabled
        if self.use_decay:
            self.current_scaling *= self.decay
            
        self.step_count += 1
        return reward
        
    def reset_scaling(self):
        """Reset scaling factor to initial value."""
        self.current_scaling = self.initial_scaling
        self.step_count = 0

@register_reward_component
class ProgressReward(BaseReward):
    """
    Reward component that tracks progress toward a goal.
    Provides dense reward signal based on distance to goal.
    """
    
    def __init__(self,
                 goal_key: str = "goal",
                 state_key: str = "state",
                 scaling: float = 1.0,
                 distance_metric: str = "euclidean",
                 normalize: bool = True,
                 **kwargs):
        """
        Initialize progress reward component.
        
        Args:
            goal_key: Key in context containing goal state/position
            state_key: Key in context containing current state/position
            scaling: Scaling factor for the reward
            distance_metric: Distance metric ("euclidean", "manhattan", "cosine")
            normalize: Whether to normalize the reward
        """
        super().__init__(**kwargs)
        self.goal_key = goal_key
        self.state_key = state_key
        self.scaling = scaling
        self.distance_metric = distance_metric
        self.normalize = normalize
        
        # Track previous distance for progress calculation
        self.previous_distance = None
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate progress-based reward."""
        # Extract goal and current state
        goal = context.get(self.goal_key)
        state = context.get(self.state_key)
        
        if goal is None or state is None:
            return 0.0
            
        # Convert to numpy arrays
        try:
            goal = np.array(goal)
            state = np.array(state)
        except (ValueError, TypeError):
            return 0.0
            
        # Calculate distance to goal
        current_distance = self._calculate_distance(state, goal)
        
        # Calculate progress reward
        if self.previous_distance is None:
            # First step - reward based on negative distance
            reward = -current_distance
        else:
            # Progress reward - positive if getting closer
            progress = self.previous_distance - current_distance
            reward = progress
            
        # Update previous distance
        self.previous_distance = current_distance
        
        # Apply scaling
        reward *= self.scaling
        
        # Normalize if requested
        if self.normalize and current_distance > 0:
            reward = reward / (current_distance + 1.0)
            
        return reward
        
    def _calculate_distance(self, state: np.ndarray, goal: np.ndarray) -> float:
        """Calculate distance between state and goal."""
        if self.distance_metric == "euclidean":
            return np.linalg.norm(state - goal)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(state - goal))
        elif self.distance_metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            dot_product = np.dot(state, goal)
            norms = np.linalg.norm(state) * np.linalg.norm(goal)
            if norms == 0:
                return 1.0
            cosine_sim = dot_product / norms
            return 1.0 - cosine_sim
        else:
            return np.linalg.norm(state - goal)  # Default to euclidean

@register_reward_component
class CoherenceReward(BaseReward):
    """
    Reward component that measures coherence or consistency in responses/actions.
    Useful for language models and sequential decision making.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 history_key: str = "history",
                 coherence_metric: str = "semantic",
                 window_size: int = 5,
                 strength: float = 1.0,
                 **kwargs):
        """
        Initialize coherence reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            history_key: Key in context containing conversation history
            coherence_metric: Method to measure coherence ("semantic", "lexical", "structural")
            window_size: Size of context window for coherence calculation
            strength: Scaling factor for the reward
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.history_key = history_key
        self.coherence_metric = coherence_metric
        self.window_size = window_size
        self.strength = strength
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate coherence-based reward."""
        text = context.get(self.text_key, "")
        history = context.get(self.history_key, [])
        
        if not text or not isinstance(text, str):
            return 0.0
            
        # Calculate coherence based on metric
        if self.coherence_metric == "semantic":
            coherence = self._calculate_semantic_coherence(text, history)
        elif self.coherence_metric == "lexical":
            coherence = self._calculate_lexical_coherence(text, history)
        elif self.coherence_metric == "structural":
            coherence = self._calculate_structural_coherence(text)
        else:
            coherence = self._calculate_lexical_coherence(text, history)
            
        return self.strength * coherence
        
    def _calculate_semantic_coherence(self, text: str, history: List[str]) -> float:
        """Calculate semantic coherence (simplified version)."""
        # This is a simplified implementation
        # In practice, you'd use embeddings or language models
        
        if not history:
            return 0.5  # Neutral score for no history
            
        # Simple word overlap as proxy for semantic similarity
        text_words = set(text.lower().split())
        
        # Get recent history
        recent_history = history[-self.window_size:]
        
        total_overlap = 0.0
        for hist_text in recent_history:
            if isinstance(hist_text, str):
                hist_words = set(hist_text.lower().split())
                if hist_words:
                    overlap = len(text_words.intersection(hist_words)) / len(hist_words.union(text_words))
                    total_overlap += overlap
                    
        return total_overlap / len(recent_history) if recent_history else 0.0
        
    def _calculate_lexical_coherence(self, text: str, history: List[str]) -> float:
        """Calculate lexical coherence based on word repetition patterns."""
        words = text.lower().split()
        if len(words) < 2:
            return 0.0
            
        # Calculate word repetition ratio
        unique_words = len(set(words))
        total_words = len(words)
        
        # Coherence is balance between repetition and diversity
        repetition_ratio = 1.0 - (unique_words / total_words)
        
        # Optimal repetition ratio is around 0.3-0.5 for coherent text
        optimal_ratio = 0.4
        coherence = 1.0 - abs(repetition_ratio - optimal_ratio) / optimal_ratio
        
        return max(0.0, coherence)
        
    def _calculate_structural_coherence(self, text: str) -> float:
        """Calculate structural coherence based on sentence structure."""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
            
        # Simple structural coherence: consistent sentence length
        lengths = [len(s.split()) for s in sentences]
        mean_length = np.mean(lengths)
        
        if mean_length == 0:
            return 0.0
            
        # Coherence based on length consistency
        variance = np.var(lengths)
        coherence = 1.0 / (1.0 + variance / mean_length)
        
        return coherence

@register_reward_component
class RelevanceReward(BaseReward):
    """
    Reward component that measures relevance to a query or context.
    """
    
    def __init__(self,
                 query_key: str = "query",
                 response_key: str = "response",
                 relevance_metric: str = "keyword_overlap",
                 strength: float = 1.0,
                 **kwargs):
        """
        Initialize relevance reward component.
        
        Args:
            query_key: Key in context containing the query/prompt
            response_key: Key in context containing the response
            relevance_metric: Method to measure relevance ("keyword_overlap", "semantic")
            strength: Scaling factor for the reward
        """
        super().__init__(**kwargs)
        self.query_key = query_key
        self.response_key = response_key
        self.relevance_metric = relevance_metric
        self.strength = strength
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate relevance-based reward."""
        query = context.get(self.query_key, "")
        response = context.get(self.response_key, "")
        
        if not query or not response:
            return 0.0
            
        # Calculate relevance based on metric
        if self.relevance_metric == "keyword_overlap":
            relevance = self._calculate_keyword_overlap(query, response)
        elif self.relevance_metric == "semantic":
            relevance = self._calculate_semantic_relevance(query, response)
        else:
            relevance = self._calculate_keyword_overlap(query, response)
            
        return self.strength * relevance
        
    def _calculate_keyword_overlap(self, query: str, response: str) -> float:
        """Calculate relevance based on keyword overlap."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = query_words.intersection(response_words)
        union = query_words.union(response_words)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _calculate_semantic_relevance(self, query: str, response: str) -> float:
        """Calculate semantic relevance (simplified implementation)."""
        # This is a placeholder for more sophisticated semantic similarity
        # In practice, you'd use sentence embeddings or language models
        
        # For now, use enhanced keyword matching with weights
        query_words = query.lower().split()
        response_words = response.lower().split()
        
        if not query_words:
            return 0.0
            
        # Weight words by inverse frequency (simple TF-IDF approximation)
        word_counts = defaultdict(int)
        for word in query_words + response_words:
            word_counts[word] += 1
            
        total_words = len(query_words) + len(response_words)
        
        relevance_score = 0.0
        for word in query_words:
            if word in response_words:
                # Weight by inverse frequency
                weight = 1.0 / (word_counts[word] / total_words + 0.01)
                relevance_score += weight
                
        # Normalize by query length
        return relevance_score / len(query_words)

@register_reward_component
class SafetyReward(BaseReward):
    """
    Reward component that penalizes unsafe or harmful content.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 safety_keywords: Optional[List[str]] = None,
                 penalty_strength: float = -2.0,
                 safety_threshold: float = 0.1,
                 **kwargs):
        """
        Initialize safety reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            safety_keywords: List of keywords that indicate unsafe content
            penalty_strength: Penalty for unsafe content (should be negative)
            safety_threshold: Threshold for safety scoring
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.penalty_strength = penalty_strength
        self.safety_threshold = safety_threshold
        
        # Default safety keywords (basic list)
        if safety_keywords is None:
            self.safety_keywords = [
                "violence", "harm", "kill", "hurt", "attack", "weapon",
                "hate", "discrimination", "racist", "sexist",
                "illegal", "drugs", "steal", "fraud"
            ]
        else:
            self.safety_keywords = safety_keywords
            
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate safety-based reward."""
        text = context.get(self.text_key, "")
        
        if not text or not isinstance(text, str):
            return 0.0
            
        # Calculate safety score
        safety_score = self._calculate_safety_score(text)
        
        # Apply penalty if below threshold
        if safety_score < self.safety_threshold:
            return self.penalty_strength * (self.safety_threshold - safety_score)
        else:
            return 0.0  # No penalty for safe content
            
    def _calculate_safety_score(self, text: str) -> float:
        """Calculate safety score (1.0 = completely safe, 0.0 = unsafe)."""
        text_lower = text.lower()
        
        # Count safety keyword occurrences
        keyword_count = 0
        for keyword in self.safety_keywords:
            keyword_count += text_lower.count(keyword.lower())
            
        # Calculate safety score based on keyword density
        word_count = len(text.split())
        if word_count == 0:
            return 1.0
            
        keyword_density = keyword_count / word_count
        
        # Safety score decreases with keyword density
        safety_score = max(0.0, 1.0 - keyword_density * 10)  # Scale factor of 10
        
        return safety_score
