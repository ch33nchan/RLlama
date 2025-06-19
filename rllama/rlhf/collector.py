# rllama/rlhf/collector.py

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import time
import random
from collections import deque

class PreferenceCollector:
    """Collects and manages human preference data."""
    
    def __init__(self, 
                buffer_size: int = 10000,
                sampling_strategy: str = 'random'):
        """
        Initialize preference collector.
        
        Args:
            buffer_size: Maximum number of preferences to store
            sampling_strategy: How to sample preferences ('random', 'recent', 'uncertainty')
        """
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        
        # Preferences buffer
        self.states_a = []
        self.states_b = []
        self.preferences = []  # 1.0 if A > B, 0.5 if A = B, 0.0 if A < B
        self.metadata = []  # Additional info about each sample
        
        # Keep track of buffer position
        self.position = 0
        self.buffer_full = False
        
    def add_preference(self, 
                      state_a: np.ndarray,
                      state_b: np.ndarray,
                      preference: float,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a preference to the buffer.
        
        Args:
            state_a: First state
            state_b: Second state
            preference: 1.0 if A > B, 0.5 if A = B, 0.0 if A < B
            metadata: Additional information about this preference
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not provided
        if 'timestamp' not in metadata:
            metadata['timestamp'] = time.time()
            
        # Check if buffer is full
        if self.buffer_full:
            # Replace existing data at position
            self.states_a[self.position] = state_a
            self.states_b[self.position] = state_b
            self.preferences[self.position] = preference
            self.metadata[self.position] = metadata
        else:
            # Add new data
            self.states_a.append(state_a)
            self.states_b.append(state_b)
            self.preferences.append(preference)
            self.metadata.append(metadata)
            
        # Update position
        self.position = (self.position + 1) % self.buffer_size
        
        # Check if buffer is now full
        if len(self.states_a) == self.buffer_size:
            self.buffer_full = True
            
    def add_symmetric_preference(self,
                               state_a: np.ndarray,
                               state_b: np.ndarray,
                               preference: float,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a preference and its symmetric counterpart.
        
        Args:
            state_a: First state
            state_b: Second state
            preference: 1.0 if A > B, 0.5 if A = B, 0.0 if A < B
            metadata: Additional information about this preference
        """
        # Add original preference
        self.add_preference(state_a, state_b, preference, metadata)
        
        # Add symmetric preference if not a tie
        if preference != 0.5:
            # Create symmetric metadata
            if metadata is None:
                sym_metadata = {}
            else:
                sym_metadata = metadata.copy()
                sym_metadata['symmetric'] = True
                
            # Add symmetric preference (invert preference value)
            self.add_preference(state_b, state_a, 1.0 - preference, sym_metadata)
            
    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of preferences.
        
        Args:
            batch_size: Number of preferences to sample
            
        Returns:
            Tuple of (states_a, states_b, preferences)
        """
        # Limit batch size to buffer size
        actual_size = min(batch_size, len(self.states_a))
        
        if self.sampling_strategy == 'recent':
            # Sample the most recent preferences
            if self.buffer_full:
                # Buffer is full, sample from the end wrapping around
                indices = [(self.position - i - 1) % self.buffer_size 
                         for i in range(actual_size)]
            else:
                # Buffer not full, sample from the end
                indices = list(range(len(self.states_a) - actual_size, len(self.states_a)))
                
        elif self.sampling_strategy == 'uncertainty':
            # Sample based on uncertainty (requires metadata with 'uncertainty')
            if all('uncertainty' in meta for meta in self.metadata):
                # Get uncertainty values
                uncertainties = [meta.get('uncertainty', 0.0) for meta in self.metadata]
                
                # Convert to sampling probabilities
                probs = np.array(uncertainties) / sum(uncertainties)
                indices = np.random.choice(
                    len(self.states_a), size=actual_size, replace=False, p=probs)
            else:
                # Fall back to random sampling
                indices = np.random.choice(len(self.states_a), size=actual_size, replace=False)
                
        else:  # 'random' or fallback
            # Sample uniformly at random
            indices = np.random.choice(len(self.states_a), size=actual_size, replace=False)
            
        # Extract samples
        batch_a = np.array([self.states_a[i] for i in indices])
        batch_b = np.array([self.states_b[i] for i in indices])
        batch_prefs = np.array([self.preferences[i] for i in indices])
        
        return batch_a, batch_b, batch_prefs
        
    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all preference data.
        
        Returns:
            Tuple of (states_a, states_b, preferences)
        """
        return (
            np.array(self.states_a),
            np.array(self.states_b),
            np.array(self.preferences)
        )
        
    def save(self, path: str) -> None:
        """
        Save preference data to a file.
        
        Args:
            path: File path to save to
        """
        data = {
            'states_a': self.states_a,
            'states_b': self.states_b,
            'preferences': self.preferences,
            'metadata': self.metadata,
            'buffer_size': self.buffer_size,
            'position': self.position,
            'buffer_full': self.buffer_full,
            'sampling_strategy': self.sampling_strategy
        }
        
        np.save(path, data, allow_pickle=True)
        
    @classmethod
    def load(cls, path: str) -> 'PreferenceCollector':
        """
        Load preference data from a file.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded PreferenceCollector instance
        """
        data = np.load(path, allow_pickle=True).item()
        
        collector = cls(
            buffer_size=data['buffer_size'],
            sampling_strategy=data['sampling_strategy']
        )
        
        collector.states_a = data['states_a']
        collector.states_b = data['states_b']
        collector.preferences = data['preferences']
        collector.metadata = data['metadata']
        collector.position = data['position']
        collector.buffer_full = data['buffer_full']
        
        return collector


class ActivePreferenceCollector(PreferenceCollector):
    """
    Preference collector with active learning capabilities.
    Uses model uncertainty to select the most informative queries.
    """
    
    def __init__(self, 
                buffer_size: int = 10000,
                sampling_strategy: str = 'uncertainty',
                model: Optional[Any] = None,
                query_batch_size: int = 100):
        """
        Initialize active preference collector.
        
        Args:
            buffer_size: Maximum number of preferences to store
            sampling_strategy: How to sample preferences
            model: Optional model to compute uncertainty estimates
            query_batch_size: Batch size for selecting queries
        """
        super().__init__(buffer_size, sampling_strategy)
        self.model = model
        self.query_batch_size = query_batch_size
        
        # Keep candidate states for querying
        self.candidate_states = []
        
    def add_candidate_states(self, states: List[np.ndarray]) -> None:
        """
        Add candidate states for querying.
        
        Args:
            states: List of state arrays
        """
        self.candidate_states.extend(states)
        
    def clear_candidates(self) -> None:
        """Clear candidate states."""
        self.candidate_states = []
        
    def select_query_pair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the most informative pair of states to query.
        
        Returns:
            Tuple of (state_a, state_b)
        """
        if not self.candidate_states or self.model is None:
            # If no candidates or model, select random pair
            if len(self.candidate_states) >= 2:
                indices = np.random.choice(len(self.candidate_states), size=2, replace=False)
                return self.candidate_states[indices[0]], self.candidate_states[indices[1]]
            else:
                # Not enough candidates
                return None, None
                
        # Use at most query_batch_size candidates
        if len(self.candidate_states) > self.query_batch_size:
            batch_indices = np.random.choice(
                len(self.candidate_states), size=self.query_batch_size, replace=False)
            batch_states = [self.candidate_states[i] for i in batch_indices]
        else:
            batch_states = self.candidate_states
            
        # Convert to tensor for model
        import torch
        states_tensor = torch.FloatTensor(np.array(batch_states))
        
        # Compute rewards and uncertainties
        with torch.no_grad():
            if hasattr(self.model, 'return_uncertainty') and callable(getattr(self.model, 'return_uncertainty')):
                # Model has built-in uncertainty estimation
                rewards, uncertainties = self.model(states_tensor, return_uncertainty=True)
                rewards = rewards.numpy()
                uncertainties = uncertainties.numpy()
            else:
                # Default to using reward magnitude as proxy for uncertainty
                rewards = self.model(states_tensor).numpy()
                uncertainties = np.abs(rewards)
                
        # Find pair with highest expected information gain
        # (proxy: pair with similar rewards but high uncertainty)
        best_score = -float('inf')
        best_pair = (0, 0)
        
        n_states = len(batch_states)
        for i in range(n_states):
            for j in range(i + 1, n_states):
                # Score based on reward difference and uncertainty
                reward_diff = abs(rewards[i] - rewards[j])
                uncertainty_sum = uncertainties[i] + uncertainties[j]
                
                # We want pairs that are:
                # - Close in reward (small reward_diff)
                # - High in uncertainty (large uncertainty_sum)
                # The most informative pairs are those where the model is uncertain
                # about which one is better
                informativeness = uncertainty_sum / (reward_diff + 1e-6)
                
                if informativeness > best_score:
                    best_score = informativeness
                    best_pair = (i, j)
                    
        # Get the selected states
        i, j = best_pair
        state_a = batch_states[i]
        state_b = batch_states[j]
        
        return state_a, state_b
