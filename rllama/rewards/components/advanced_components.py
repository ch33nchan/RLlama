# rllama/rewards/components/advanced_components.py

from typing import Dict, Any, List, Set
import numpy as np
from datetime import datetime

from rllama.rewards.base import BaseReward  # Fixed import path

class ExplorationReward(BaseReward):
    """
    Rewards the agent for exploring new states.
    States visited for the first time receive a higher reward than repeat visits.
    """
    
    def __init__(self, reward_scale: float = 0.1):
        """
        Initialize the exploration reward component.
        
        Args:
            reward_scale: Scale factor for the reward
        """
        super().__init__()
        self.visited_states = set()
        self.visit_counts = {}
        self.reward_scale = reward_scale
    
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate exploration reward based on state novelty.
        
        Args:
            context: Must contain a 'state_hash' key with a hashable representation 
                     of the current state
                     
        Returns:
            Exploration reward value
        """
        # Get state hash from context
        state_hash = context.get('state_hash')
        
        if state_hash is None:
            return 0.0
            
        # Get current visit count
        visit_count = self.visit_counts.get(state_hash, 0)
        
        # Update visit count
        self.visit_counts[state_hash] = visit_count + 1
        
        # Add to visited states
        self.visited_states.add(state_hash)
        
        # Calculate reward (higher for first visits, decays with repeat visits)
        if visit_count == 0:
            # First visit
            return self.reward_scale
        else:
            # Repeat visit - decay reward
            return self.reward_scale / (visit_count + 1)
    
    def reset(self) -> None:
        """Reset the visited states tracker"""
        self.visited_states = set()
        self.visit_counts = {}


class DiversityReward(BaseReward):
    """
    Rewards the agent for producing diverse behaviors or outputs.
    Higher rewards are given for actions or outputs that differ from recent history.
    """
    
    def __init__(self, target_key: str = 'action', history_size: int = 5, reward_scale: float = 0.1):
        """
        Initialize the diversity reward component.
        
        Args:
            target_key: The key in the context to measure diversity for
            history_size: How many past items to keep in history
            reward_scale: Scale factor for the reward
        """
        super().__init__()
        self.target_key = target_key
        self.history_size = history_size
        self.history = []
        self.reward_scale = reward_scale
    
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate diversity reward based on how different the current target is from history.
        
        Args:
            context: Must contain the target_key that was specified in __init__
                     
        Returns:
            Diversity reward value
        """
        # Get target value from context
        target_value = context.get(self.target_key)
        
        if target_value is None:
            return 0.0
            
        # Calculate how common this value is in history
        if not self.history:
            # First item, maximum diversity
            diversity_score = 1.0
        else:
            # Count occurrences in history
            occurrences = sum(1 for x in self.history if x == target_value)
            
            if occurrences == 0:
                # Not in history, maximum diversity
                diversity_score = 1.0
            else:
                # In history, lower diversity score
                diversity_score = 1.0 / (occurrences + 1)
        
        # Update history
        self.history.append(target_value)
        
        # Keep history at desired size
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
        # Return scaled reward
        return self.reward_scale * diversity_score
    
    def reset(self) -> None:
        """Reset the history"""
        self.history = []


class NoveltyReward(BaseReward):
    """
    Rewards the agent for generating novel content based on some similarity metric.
    Novelty is calculated based on similarity to a reference corpus.
    """
    
    def __init__(self, 
                 corpus: List[Any] = None,
                 similarity_threshold: float = 0.7,
                 reward_scale: float = 0.2):
        """
        Initialize the novelty reward component.
        
        Args:
            corpus: Reference corpus to compare against
            similarity_threshold: Threshold below which content is considered novel
            reward_scale: Scale factor for the reward
        """
        super().__init__()
        self.corpus = corpus or []
        self.similarity_threshold = similarity_threshold
        self.reward_scale = reward_scale
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate novelty reward based on similarity to corpus.
        
        Args:
            context: Must contain 'content' and 'similarity_fn' keys
                    - content: The content to evaluate
                    - similarity_fn: A function that computes similarity between two items
                     
        Returns:
            Novelty reward value
        """
        content = context.get('content')
        similarity_fn = context.get('similarity_fn')
        
        if content is None or similarity_fn is None or not self.corpus:
            return 0.0
            
        # Calculate maximum similarity to corpus
        max_similarity = max(
            similarity_fn(content, corpus_item) 
            for corpus_item in self.corpus
        )
        
        # Calculate novelty (1 - similarity)
        novelty = 1.0 - max_similarity
        
        # Only reward if novelty is above threshold
        if novelty > (1.0 - self.similarity_threshold):
            return self.reward_scale * novelty
        
        return 0.0
    
    def add_to_corpus(self, item: Any) -> None:
        """Add an item to the reference corpus"""
        self.corpus.append(item)
    
    def reset(self) -> None:
        """Reset the corpus"""
        self.corpus = []
