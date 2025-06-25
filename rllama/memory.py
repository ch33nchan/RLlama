#!/usr/bin/env python3
"""
Advanced memory systems for RLlama framework.
Provides episodic memory, working memory, and memory compression capabilities.
"""

import torch
import numpy as np
import time
import pickle
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from collections import deque, defaultdict
import heapq
import logging
from pathlib import Path
import warnings

@dataclass
class MemoryEntry:
    """
    A comprehensive memory entry for episodic memory storage.
    Contains state, action, reward, and metadata information.
    """
    state: Union[torch.Tensor, np.ndarray, List[float]]
    action: Any
    reward: float
    next_state: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None
    done: bool = False
    timestamp: float = 0.0
    importance: float = 0.0
    episode_id: Optional[str] = None
    step_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary format."""
        result = asdict(self)
        
        # Convert tensors to lists for serialization
        if isinstance(self.state, torch.Tensor):
            result['state'] = self.state.tolist()
        elif isinstance(self.state, np.ndarray):
            result['state'] = self.state.tolist()
            
        if self.next_state is not None:
            if isinstance(self.next_state, torch.Tensor):
                result['next_state'] = self.next_state.tolist()
            elif isinstance(self.next_state, np.ndarray):
                result['next_state'] = self.next_state.tolist()
                
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory entry from dictionary."""
        # Convert lists back to tensors if needed
        if isinstance(data.get('state'), list):
            data['state'] = torch.FloatTensor(data['state'])
        if isinstance(data.get('next_state'), list):
            data['next_state'] = torch.FloatTensor(data['next_state'])
            
        return cls(**data)

class EpisodicMemory:
    """
    Advanced episodic memory storage for reinforcement learning.
    Stores experiences with sophisticated retrieval and management capabilities.
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 similarity_metric: str = "cosine",
                 importance_decay: float = 0.99,
                 auto_compress: bool = True,
                 compression_threshold: float = 0.8):
        """
        Initialize episodic memory.
        
        Args:
            capacity: Maximum number of memories to store
            similarity_metric: Metric for similarity calculation ("cosine", "euclidean", "dot")
            importance_decay: Decay factor for importance scores over time
            auto_compress: Whether to automatically compress similar memories
            compression_threshold: Threshold for automatic compression
        """
        self.capacity = capacity
        self.similarity_metric = similarity_metric
        self.importance_decay = importance_decay
        self.auto_compress = auto_compress
        self.compression_threshold = compression_threshold
        
        # Memory storage
        self.memories: List[MemoryEntry] = []
        self.next_idx = 0
        self.total_added = 0
        
        # Indexing for fast retrieval
        self.episode_index: Dict[str, List[int]] = defaultdict(list)
        self.importance_heap: List[Tuple[float, int]] = []  # Min-heap for importance
        
        # Statistics
        self.access_counts: Dict[int, int] = defaultdict(int)
        self.last_access_times: Dict[int, float] = {}
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def add(self, memory_entry: MemoryEntry) -> None:
        """
        Add a new memory entry with sophisticated management.
        
        Args:
            memory_entry: The memory entry to add
        """
        # Set importance if not provided
        if memory_entry.importance == 0.0:
            memory_entry.importance = self._calculate_initial_importance(memory_entry)
            
        # Check for similar memories if auto-compression is enabled
        if self.auto_compress and len(self.memories) > 0:
            similar_idx = self._find_most_similar(memory_entry)
            if similar_idx is not None:
                similarity = self._calculate_similarity(
                    memory_entry, self.memories[similar_idx]
                )
                if similarity > self.compression_threshold:
                    # Merge with existing memory instead of adding new one
                    self._merge_memories(similar_idx, memory_entry)
                    return
        
        # Add memory to storage
        if len(self.memories) < self.capacity:
            # Still have space
            idx = len(self.memories)
            self.memories.append(memory_entry)
        else:
            # Need to replace existing memory
            idx = self._select_replacement_index()
            old_memory = self.memories[idx]
            
            # Update episode index
            if old_memory.episode_id:
                if idx in self.episode_index[old_memory.episode_id]:
                    self.episode_index[old_memory.episode_id].remove(idx)
                    
            self.memories[idx] = memory_entry
            
        # Update episode index
        if memory_entry.episode_id:
            self.episode_index[memory_entry.episode_id].append(idx)
            
        # Update importance heap
        heapq.heappush(self.importance_heap, (memory_entry.importance, idx))
        
        # Update statistics
        self.total_added += 1
        self.last_access_times[idx] = time.time()
        
    def retrieve_relevant(self, 
                         query: Union[torch.Tensor, np.ndarray, List[float]], 
                         k: int = 5,
                         filter_fn: Optional[Callable[[MemoryEntry], bool]] = None,
                         boost_recent: bool = True) -> List[Tuple[MemoryEntry, float]]:
        """
        Retrieve the k most relevant memories with similarity scores.
        
        Args:
            query: Query state/embedding to find similar memories for
            k: Number of memories to retrieve
            filter_fn: Optional function to filter memories
            boost_recent: Whether to boost scores for recent memories
            
        Returns:
            List of (memory_entry, similarity_score) tuples
        """
        if not self.memories:
            return []
            
        # Convert query to tensor if needed
        if isinstance(query, list):
            query = torch.FloatTensor(query)
        elif isinstance(query, np.ndarray):
            query = torch.from_numpy(query).float()
            
        similarities = []
        current_time = time.time()
        
        for i, memory in enumerate(self.memories):
            # Apply filter if provided
            if filter_fn and not filter_fn(memory):
                continue
                
            # Calculate similarity
            similarity = self._calculate_similarity_to_query(query, memory)
            
            if similarity is None:
                continue
                
            # Boost recent memories if requested
            if boost_recent:
                time_factor = math.exp(-(current_time - memory.timestamp) / 3600)  # 1-hour decay
                similarity *= (1.0 + 0.1 * time_factor)
                
            # Boost important memories
            importance_factor = 1.0 + 0.1 * memory.importance
            similarity *= importance_factor
            
            similarities.append((similarity, i, memory))
            
        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Update access statistics
        result = []
        for similarity, idx, memory in similarities[:k]:
            self.access_counts[idx] += 1
            self.last_access_times[idx] = current_time
            result.append((memory, similarity))
            
        return result
        
    def retrieve_by_episode(self, episode_id: str) -> List[MemoryEntry]:
        """
        Retrieve all memories from a specific episode.
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            List of memory entries from the episode
        """
        if episode_id not in self.episode_index:
            return []
            
        indices = self.episode_index[episode_id]
        return [self.memories[idx] for idx in indices if idx < len(self.memories)]
        
    def retrieve_by_importance(self, k: int = 5, min_importance: float = 0.0) -> List[MemoryEntry]:
        """
        Retrieve memories by importance score.
        
        Args:
            k: Number of memories to retrieve
            min_importance: Minimum importance threshold
            
        Returns:
            List of most important memory entries
        """
        # Get all memories with importance above threshold
        important_memories = [
            (memory.importance, memory) 
            for memory in self.memories 
            if memory.importance >= min_importance
        ]
        
        # Sort by importance and return top k
        important_memories.sort(reverse=True, key=lambda x: x[0])
        return [memory for _, memory in important_memories[:k]]
        
    def update_importance(self, 
                         memory_filter: Callable[[MemoryEntry], bool],
                         importance_update: Callable[[MemoryEntry], float]) -> int:
        """
        Update importance scores for memories matching a filter.
        
        Args:
            memory_filter: Function to filter which memories to update
            importance_update: Function to calculate new importance
            
        Returns:
            Number of memories updated
        """
        updated_count = 0
        
        for memory in self.memories:
            if memory_filter(memory):
                old_importance = memory.importance
                memory.importance = importance_update(memory)
                updated_count += 1
                
                self.logger.debug(f"Updated importance: {old_importance:.3f} -> {memory.importance:.3f}")
                
        # Rebuild importance heap
        self.importance_heap = [
            (memory.importance, i) 
            for i, memory in enumerate(self.memories)
        ]
        heapq.heapify(self.importance_heap)
        
        return updated_count
        
    def decay_importance(self, decay_factor: Optional[float] = None) -> None:
        """
        Apply importance decay to all memories.
        
        Args:
            decay_factor: Decay factor to use (uses instance default if None)
        """
        if decay_factor is None:
            decay_factor = self.importance_decay
            
        for memory in self.memories:
            memory.importance *= decay_factor
            
        # Rebuild importance heap
        self.importance_heap = [
            (memory.importance, i) 
            for i, memory in enumerate(self.memories)
        ]
        heapq.heapify(self.importance_heap)
        
    def compress_similar_memories(self, similarity_threshold: float = 0.9) -> int:
        """
        Compress similar memories to save space.
        
        Args:
            similarity_threshold: Threshold for considering memories similar
            
        Returns:
            Number of memories compressed
        """
        if len(self.memories) < 2:
            return 0
            
        compressed_count = 0
        to_remove = set()
        
        for i in range(len(self.memories)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(self.memories)):
                if j in to_remove:
                    continue
                    
                similarity = self._calculate_similarity(self.memories[i], self.memories[j])
                
                if similarity is not None and similarity > similarity_threshold:
                    # Merge j into i
                    self._merge_memories(i, self.memories[j])
                    to_remove.add(j)
                    compressed_count += 1
                    
        # Remove compressed memories
        if to_remove:
            self.memories = [
                memory for i, memory in enumerate(self.memories) 
                if i not in to_remove
            ]
            
            # Rebuild indices
            self._rebuild_indices()
            
        return compressed_count
        
    def save(self, path: Union[str, Path]) -> None:
        """
        Save episodic memory to file.
        
        Args:
            path: Path to save the memory
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert memories to serializable format
        serializable_memories = [memory.to_dict() for memory in self.memories]
        
        data = {
            'memories': serializable_memories,
            'capacity': self.capacity,
            'similarity_metric': self.similarity_metric,
            'importance_decay': self.importance_decay,
            'auto_compress': self.auto_compress,
            'compression_threshold': self.compression_threshold,
            'total_added': self.total_added,
            'access_counts': dict(self.access_counts),
            'last_access_times': self.last_access_times
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Saved {len(self.memories)} memories to {path}")
        
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EpisodicMemory':
        """
        Load episodic memory from file.
        
        Args:
            path: Path to load the memory from
            
        Returns:
            Loaded EpisodicMemory instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Create instance
        memory = cls(
            capacity=data['capacity'],
            similarity_metric=data['similarity_metric'],
            importance_decay=data['importance_decay'],
            auto_compress=data['auto_compress'],
            compression_threshold=data['compression_threshold']
        )
        
        # Load memories
        memory.memories = [MemoryEntry.from_dict(mem_data) for mem_data in data['memories']]
        memory.total_added = data['total_added']
        memory.access_counts = defaultdict(int, data['access_counts'])
        memory.last_access_times = data['last_access_times']
        
        # Rebuild indices
        memory._rebuild_indices()
        
        return memory
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        if not self.memories:
            return {"total_memories": 0}
            
        importances = [memory.importance for memory in self.memories]
        timestamps = [memory.timestamp for memory in self.memories]
        current_time = time.time()
        
        return {
            "total_memories": len(self.memories),
            "capacity": self.capacity,
            "capacity_utilization": len(self.memories) / self.capacity,
            "total_added": self.total_added,
            "average_importance": np.mean(importances),
            "max_importance": np.max(importances),
            "min_importance": np.min(importances),
            "average_age_hours": np.mean([(current_time - ts) / 3600 for ts in timestamps]),
            "unique_episodes": len(self.episode_index),
            "total_accesses": sum(self.access_counts.values()),
            "most_accessed_count": max(self.access_counts.values()) if self.access_counts else 0
        }
        
    def _calculate_initial_importance(self, memory_entry: MemoryEntry) -> float:
        """Calculate initial importance for a memory entry."""
        # Base importance on reward magnitude and recency
        reward_factor = abs(memory_entry.reward)
        recency_factor = 1.0  # New memories start with full recency
        
        # Bonus for terminal states
        terminal_bonus = 0.1 if memory_entry.done else 0.0
        
        return reward_factor + recency_factor + terminal_bonus
        
    def _calculate_similarity(self, memory1: MemoryEntry, memory2: MemoryEntry) -> Optional[float]:
        """Calculate similarity between two memory entries."""
        return self._calculate_similarity_between_states(memory1.state, memory2.state)
        
    def _calculate_similarity_to_query(self, 
                                     query: torch.Tensor, 
                                     memory: MemoryEntry) -> Optional[float]:
        """Calculate similarity between query and memory state."""
        return self._calculate_similarity_between_states(query, memory.state)
        
    def _calculate_similarity_between_states(self, 
                                           state1: Union[torch.Tensor, np.ndarray, List[float]], 
                                           state2: Union[torch.Tensor, np.ndarray, List[float]]) -> Optional[float]:
        """Calculate similarity between two states."""
        try:
            # Convert to tensors
            if isinstance(state1, list):
                state1 = torch.FloatTensor(state1)
            elif isinstance(state1, np.ndarray):
                state1 = torch.from_numpy(state1).float()
                
            if isinstance(state2, list):
                state2 = torch.FloatTensor(state2)
            elif isinstance(state2, np.ndarray):
                state2 = torch.from_numpy(state2).float()
                
            # Ensure same shape
            if state1.shape != state2.shape:
                return None
                
            # Calculate similarity based on metric
            if self.similarity_metric == "cosine":
                dot_product = torch.sum(state1 * state2).item()
                norm1 = torch.norm(state1).item()
                norm2 = torch.norm(state2).item()
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return dot_product / (norm1 * norm2)
                
            elif self.similarity_metric == "euclidean":
                distance = torch.norm(state1 - state2).item()
                return 1.0 / (1.0 + distance)  # Convert distance to similarity
                
            elif self.similarity_metric == "dot":
                return torch.sum(state1 * state2).item()
                
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Error calculating similarity: {e}")
            return None
            
    def _find_most_similar(self, memory_entry: MemoryEntry) -> Optional[int]:
        """Find index of most similar memory."""
        if not self.memories:
            return None
            
        best_similarity = -1.0
        best_idx = None
        
        for i, existing_memory in enumerate(self.memories):
            similarity = self._calculate_similarity(memory_entry, existing_memory)
            if similarity is not None and similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
                
        return best_idx
        
    def _merge_memories(self, target_idx: int, source_memory: MemoryEntry) -> None:
        """Merge source memory into target memory."""
        target_memory = self.memories[target_idx]
        
        # Update importance (take maximum)
        target_memory.importance = max(target_memory.importance, source_memory.importance)
        
        # Update timestamp to more recent
        target_memory.timestamp = max(target_memory.timestamp, source_memory.timestamp)
        
        # Merge metadata
        if source_memory.metadata:
            if target_memory.metadata is None:
                target_memory.metadata = {}
            target_memory.metadata.update(source_memory.metadata)
            
    def _select_replacement_index(self) -> int:
        """Select index of memory to replace when at capacity."""
        # Use least important memory
        if self.importance_heap:
            _, idx = heapq.heappop(self.importance_heap)
            return idx
        else:
            # Fallback to oldest memory
            oldest_idx = 0
            oldest_time = self.memories[0].timestamp
            
            for i, memory in enumerate(self.memories):
                if memory.timestamp < oldest_time:
                    oldest_time = memory.timestamp
                    oldest_idx = i
                    
            return oldest_idx
            
    def _rebuild_indices(self) -> None:
        """Rebuild internal indices after memory modifications."""
        # Rebuild episode index
        self.episode_index.clear()
        for i, memory in enumerate(self.memories):
            if memory.episode_id:
                self.episode_index[memory.episode_id].append(i)
                
        # Rebuild importance heap
        self.importance_heap = [
            (memory.importance, i) 
            for i, memory in enumerate(self.memories)
        ]
        heapq.heapify(self.importance_heap)
        
    def __len__(self) -> int:
        """Return number of stored memories."""
        return len(self.memories)

class WorkingMemory:
    """
    Advanced working memory for temporary storage and active processing.
    Provides attention mechanisms and context generation.
    """
    
    def __init__(self, 
                 max_size: int = 10,
                 attention_mechanism: str = "dot_product",
                 decay_factor: float = 0.95):
        """
        Initialize working memory.
        
        Args:
            max_size: Maximum number of items to keep in working memory
            attention_mechanism: Type of attention ("dot_product", "scaled_dot_product", "additive")
            decay_factor: Decay factor for attention weights over time
        """
        self.max_size = max_size
        self.attention_mechanism = attention_mechanism
        self.decay_factor = decay_factor
        
        # Memory storage
        self.memory: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)
        self.attention_weights: deque = deque(maxlen=max_size)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def add(self, 
           state: Union[torch.Tensor, np.ndarray, List[float]], 
           importance: float = 1.0) -> None:
        """
        Add a state to working memory with importance weighting.
        
        Args:
            state: State or embedding to add to memory
            importance: Importance weight for the state
        """
        # Convert to tensor if needed
        if isinstance(state, list):
            state = torch.FloatTensor(state)
        elif isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            
        # Add to memory
        self.memory.append(state)
        self.timestamps.append(time.time())
        self.attention_weights.append(importance)
        
        # Decay existing attention weights
        for i in range(len(self.attention_weights) - 1):
            self.attention_weights[i] *= self.decay_factor
            
    def get_context(self, 
                   query: Union[torch.Tensor, np.ndarray, List[float]],
                   temperature: float = 1.0) -> torch.Tensor:
        """
        Generate a context vector using attention over working memory.
        
        Args:
            query: Query state or embedding
            temperature: Temperature for attention softmax
            
        Returns:
            Context vector combining query with attended memory
        """
        if not self.memory:
            # Convert query to tensor and return as-is
            if isinstance(query, list):
                return torch.FloatTensor(query)
            elif isinstance(query, np.ndarray):
                return torch.from_numpy(query).float()
            else:
                return query
                
        # Convert query to tensor
        if isinstance(query, list):
            query = torch.FloatTensor(query)
        elif isinstance(query, np.ndarray):
            query = torch.from_numpy(query).float()
            
        # Calculate attention weights
        attention_scores = []
        
        for i, memory_item in enumerate(self.memory):
            if memory_item.shape != query.shape:
                continue
                
            # Calculate attention score based on mechanism
            if self.attention_mechanism == "dot_product":
                score = torch.sum(memory_item * query).item()
            elif self.attention_mechanism == "scaled_dot_product":
                score = torch.sum(memory_item * query).item() / math.sqrt(query.numel())
            elif self.attention_mechanism == "additive":
                # Simplified additive attention
                combined = memory_item + query
                score = torch.sum(combined).item()
            else:
                score = torch.sum(memory_item * query).item()
                
            # Apply importance weighting
            score *= self.attention_weights[i]
            attention_scores.append(score)
            
        if not attention_scores:
            return query
            
        # Apply softmax with temperature
        attention_scores = torch.FloatTensor(attention_scores)
        attention_weights = torch.softmax(attention_scores / temperature, dim=0)
        
        # Compute weighted combination
        context = query.clone()
        
        valid_memories = [
            memory for memory in self.memory 
            if memory.shape == query.shape
        ]
        
        for i, (weight, memory_item) in enumerate(zip(attention_weights, valid_memories)):
            context += weight * memory_item
            
        return context
        
    def get_attention_weights(self, 
                            query: Union[torch.Tensor, np.ndarray, List[float]]) -> List[float]:
        """
        Get attention weights for visualization or analysis.
        
        Args:
            query: Query state or embedding
            
        Returns:
            List of attention weights
        """
        if not self.memory:
            return []
            
        # Convert query to tensor
        if isinstance(query, list):
            query = torch.FloatTensor(query)
        elif isinstance(query, np.ndarray):
            query = torch.from_numpy(query).float()
            
        attention_scores = []
        
        for i, memory_item in enumerate(self.memory):
            if memory_item.shape != query.shape:
                attention_scores.append(0.0)
                continue
                
            score = torch.sum(memory_item * query).item()
            score *= self.attention_weights[i]
            attention_scores.append(score)
            
        # Apply softmax
        if attention_scores:
            attention_scores = torch.FloatTensor(attention_scores)
            attention_weights = torch.softmax(attention_scores, dim=0)
            return attention_weights.tolist()
        else:
            return []
            
    def clear(self) -> None:
        """Clear all items from working memory."""
        self.memory.clear()
        self.timestamps.clear()
        self.attention_weights.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get working memory statistics."""
        if not self.memory:
            return {"size": 0, "capacity": self.max_size}
            
        current_time = time.time()
        ages = [(current_time - ts) for ts in self.timestamps]
        
        return {
            "size": len(self.memory),
            "capacity": self.max_size,
            "utilization": len(self.memory) / self.max_size,
            "average_age_seconds": np.mean(ages),
            "max_age_seconds": max(ages),
            "total_attention_weight": sum(self.attention_weights),
            "average_attention_weight": np.mean(list(self.attention_weights))
        }
        
    def __len__(self) -> int:
        """Return current size of working memory."""
        return len(self.memory)

class MemoryCompressor:
    """
    Advanced memory compression system with multiple strategies.
    Reduces memory usage while preserving important information.
    """
    
    def __init__(self, 
                 compression_ratio: float = 0.5,
                 similarity_threshold: float = 0.8,
                 importance_weight: float = 0.6,
                 recency_weight: float = 0.3,
                 diversity_weight: float = 0.1):
        """
        Initialize memory compressor.
        
        Args:
            compression_ratio: Target ratio of memories to keep (0.5 = keep 50%)
            similarity_threshold: Threshold for considering memories similar
            importance_weight: Weight for importance in compression decisions
            recency_weight: Weight for recency in compression decisions
            diversity_weight: Weight for diversity in compression decisions
        """
        self.compression_ratio = compression_ratio
        self.similarity_threshold = similarity_threshold
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight
        self.diversity_weight = diversity_weight
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def compress(self, 
                memories: List[MemoryEntry],
                strategy: str = "hybrid") -> List[MemoryEntry]:
        """
        Compress a set of memories using specified strategy.
        
        Args:
            memories: List of memory entries to compress
            strategy: Compression strategy ("importance", "similarity", "diversity", "hybrid")
            
        Returns:
            Compressed list of memory entries
        """
        if not memories:
            return []
            
        target_count = max(1, int(len(memories) * self.compression_ratio))
        
        if target_count >= len(memories):
            return memories
            
        if strategy == "importance":
            return self._compress_by_importance(memories, target_count)
        elif strategy == "similarity":
            return self._compress_by_similarity(memories, target_count)
        elif strategy == "diversity":
            return self._compress_by_diversity(memories, target_count)
        elif strategy == "hybrid":
            return self._compress_hybrid(memories, target_count)
        else:
            self.logger.warning(f"Unknown compression strategy: {strategy}")
            return self._compress_hybrid(memories, target_count)
            
    def _compress_by_importance(self, 
                               memories: List[MemoryEntry], 
                               target_count: int) -> List[MemoryEntry]:
        """Compress by keeping most important memories."""
        sorted_memories = sorted(memories, key=lambda m: m.importance, reverse=True)
        return sorted_memories[:target_count]
        
    def _compress_by_similarity(self, 
                               memories: List[MemoryEntry], 
                               target_count: int) -> List[MemoryEntry]:
        """Compress by removing similar memories."""
        if len(memories) <= target_count:
            return memories
            
        # Start with all memories
        remaining = memories.copy()
        
        while len(remaining) > target_count:
            # Find most similar pair
            max_similarity = -1.0
            most_similar_pair = None
            
            for i in range(len(remaining)):
                for j in range(i + 1, len(remaining)):
                    similarity = self._calculate_similarity(remaining[i], remaining[j])
                    if similarity is not None and similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_pair = (i, j)
                        
            if most_similar_pair is None:
                break
                
            # Remove the less important memory from the pair
            i, j = most_similar_pair
            if remaining[i].importance >= remaining[j].importance:
                remaining.pop(j)
            else:
                remaining.pop(i)
                
        return remaining
        
    def _compress_by_diversity(self, 
                              memories: List[MemoryEntry], 
                              target_count: int) -> List[MemoryEntry]:
        """Compress by maximizing diversity."""
        if len(memories) <= target_count:
            return memories
            
        # Start with most important memory
        selected = [max(memories, key=lambda m: m.importance)]
        remaining = [m for m in memories if m != selected[0]]
        
        while len(selected) < target_count and remaining:
            # Find memory that maximizes minimum distance to selected memories
            best_memory = None
            best_min_distance = -1.0
            
            for candidate in remaining:
                min_distance = float('inf')
                
                for selected_memory in selected:
                    similarity = self._calculate_similarity(candidate, selected_memory)
                    if similarity is not None:
                        distance = 1.0 - similarity
                        min_distance = min(min_distance, distance)
                        
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_memory = candidate
                    
            if best_memory is not None:
                selected.append(best_memory)
                remaining.remove(best_memory)
            else:
                break
                
        return selected
        
    def _compress_hybrid(self, 
                        memories: List[MemoryEntry], 
                        target_count: int) -> List[MemoryEntry]:
        """Compress using hybrid strategy combining multiple factors."""
        if len(memories) <= target_count:
            return memories
            
        current_time = time.time()
        
        # Calculate composite scores for all memories
        scored_memories = []
        
        for memory in memories:
            # Importance component
            importance_score = memory.importance
            
            # Recency component
            age_hours = (current_time - memory.timestamp) / 3600
            recency_score = math.exp(-age_hours / 24)  # Decay over 24 hours
            
            # Diversity component (average distance to other memories)
            diversity_score = 0.0
            count = 0
            
            for other_memory in memories:
                if other_memory != memory:
                    similarity = self._calculate_similarity(memory, other_memory)
                    if similarity is not None:
                        diversity_score += (1.0 - similarity)
                        count += 1
                        
            if count > 0:
                diversity_score /= count
                
            # Combine scores
            composite_score = (
                self.importance_weight * importance_score +
                self.recency_weight * recency_score +
                self.diversity_weight * diversity_score
            )
            
            scored_memories.append((composite_score, memory))
            
        # Sort by composite score and select top memories
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        return [memory for _, memory in scored_memories[:target_count]]
        
    def _calculate_similarity(self, 
                            memory1: MemoryEntry, 
                            memory2: MemoryEntry) -> Optional[float]:
        """Calculate similarity between two memory entries."""
        try:
            state1 = memory1.state
            state2 = memory2.state
            
            # Convert to tensors
            if isinstance(state1, list):
                state1 = torch.FloatTensor(state1)
            elif isinstance(state1, np.ndarray):
                state1 = torch.from_numpy(state1).float()
                
            if isinstance(state2, list):
                state2 = torch.FloatTensor(state2)
            elif isinstance(state2, np.ndarray):
                state2 = torch.from_numpy(state2).float()
                
            # Check shape compatibility
            if state1.shape != state2.shape:
                return None
                
            # Calculate cosine similarity
            dot_product = torch.sum(state1 * state2).item()
            norm1 = torch.norm(state1).item()
            norm2 = torch.norm(state2).item()
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.warning(f"Error calculating similarity: {e}")
            return None
            
    def get_compression_stats(self, 
                            original_memories: List[MemoryEntry],
                            compressed_memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Get statistics about compression results."""
        if not original_memories:
            return {}
            
        original_count = len(original_memories)
        compressed_count = len(compressed_memories)
        
        # Calculate importance preservation
        original_importance = sum(m.importance for m in original_memories)
        preserved_importance = sum(m.importance for m in compressed_memories)
        importance_ratio = preserved_importance / original_importance if original_importance > 0 else 0
        
        # Calculate age distribution
        current_time = time.time()
        original_ages = [(current_time - m.timestamp) / 3600 for m in original_memories]
        compressed_ages = [(current_time - m.timestamp) / 3600 for m in compressed_memories]
        
        return {
            "original_count": original_count,
            "compressed_count": compressed_count,
            "compression_ratio": compressed_count / original_count,
            "importance_preservation_ratio": importance_ratio,
            "original_avg_age_hours": np.mean(original_ages),
            "compressed_avg_age_hours": np.mean(compressed_ages),
            "original_avg_importance": np.mean([m.importance for m in original_memories]),
            "compressed_avg_importance": np.mean([m.importance for m in compressed_memories])
        }
