# rllama/memory.py

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
import heapq
from collections import deque
import time

@dataclass
class MemoryEntry:
    """A single memory entry for episodic memory"""
    state: torch.Tensor
    action: Any
    reward: float
    next_state: Optional[torch.Tensor] = None
    done: bool = False
    timestamp: int = 0
    importance: float = 0.0

class EpisodicMemory:
    """
    Episodic memory storage for reinforcement learning.
    Stores experiences and allows retrieval of relevant memories.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize episodic memory.
        
        Args:
            capacity: Maximum number of memories to store
        """
        self.capacity = capacity
        self.memories = []
        self.next_idx = 0
    
    def add(self, memory_entry: MemoryEntry) -> None:
        """
        Add a new memory entry.
        
        Args:
            memory_entry: The memory entry to add
        """
        # If we haven't reached capacity, add a new memory
        if len(self.memories) < self.capacity:
            self.memories.append(memory_entry)
        else:
            # Replace an existing memory
            self.memories[self.next_idx] = memory_entry
            
        # Update next index
        self.next_idx = (self.next_idx + 1) % self.capacity
    
    def retrieve_relevant(self, query: torch.Tensor, k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve the k most relevant memories based on state similarity.
        
        Args:
            query: Query state/embedding to find similar memories for
            k: Number of memories to retrieve
            
        Returns:
            List of the k most relevant memory entries
        """
        if not self.memories or not isinstance(query, torch.Tensor):
            return []
            
        # Calculate cosine similarity between query and all memories
        sims = []
        for i, memory in enumerate(self.memories):
            if not isinstance(memory.state, torch.Tensor):
                continue
                
            # Calculate cosine similarity: dot product / (norm(a) * norm(b))
            if memory.state.shape != query.shape:
                continue
                
            dot_product = torch.sum(memory.state * query).item()
            norm_a = torch.norm(memory.state).item()
            norm_b = torch.norm(query).item()
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_a * norm_b)
                
            sims.append((similarity, i))
        
        # Get indices of top k similar memories
        top_k = sorted(sims, reverse=True)[:k]
        
        # Return the corresponding memories
        return [self.memories[i] for _, i in top_k]
        
    def __len__(self) -> int:
        return len(self.memories)


class WorkingMemory:
    """
    Working memory for temporary storage of recent information.
    This is a smaller, more accessible memory system for active use.
    """
    
    def __init__(self, max_size: int = 5):
        """
        Initialize working memory.
        
        Args:
            max_size: Maximum number of items to keep in working memory
        """
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)
        
    def add(self, state: torch.Tensor) -> None:
        """
        Add a state to working memory.
        
        Args:
            state: State or embedding to add to memory
        """
        self.memory.append(state)
        
    def get_context(self, query: torch.Tensor) -> torch.Tensor:
        """
        Generate a context vector based on the query and current working memory.
        
        Args:
            query: Query state or embedding
            
        Returns:
            A context vector (e.g., summing query with weighted memory items)
        """
        if not self.memory or not isinstance(query, torch.Tensor):
            return query
            
        # Calculate attention weights for memory items
        weights = []
        for memory_item in self.memory:
            if not isinstance(memory_item, torch.Tensor) or memory_item.shape != query.shape:
                continue
                
            # Calculate simple dot product attention
            weight = torch.sum(memory_item * query).item()
            weights.append(weight)
            
        # Normalize weights with softmax
        if weights:
            weights = np.exp(weights) / np.sum(np.exp(weights))
            
            # Combine memory items with attention weights
            context = query.clone()
            for i, (weight, memory_item) in enumerate(zip(weights, self.memory)):
                if isinstance(memory_item, torch.Tensor) and memory_item.shape == query.shape:
                    context += weight * memory_item
                
            return context
        else:
            return query
    
    def __len__(self) -> int:
        return len(self.memory)


class MemoryCompressor:
    """
    Compresses a set of memories by clustering similar ones.
    Useful for removing redundancy in episodic memory.
    """
    
    def __init__(self, compression_ratio: float = 0.5, similarity_threshold: float = 0.8):
        """
        Initialize memory compressor.
        
        Args:
            compression_ratio: Target ratio of memories to keep (0.5 means keep 50%)
            similarity_threshold: Threshold above which memories are considered similar
        """
        self.compression_ratio = compression_ratio
        self.similarity_threshold = similarity_threshold
        
    def compress(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Compress a set of memories.
        
        Args:
            memories: List of memory entries to compress
            
        Returns:
            Compressed list of memory entries
        """
        if not memories:
            return []
            
        # Target number of memories after compression
        target_count = max(1, int(len(memories) * self.compression_ratio))
        
        # If target count is already greater than or equal to current memories, no compression needed
        if target_count >= len(memories):
            return memories
            
        # Sort memories by importance
        sorted_by_importance = sorted(memories, key=lambda x: x.importance, reverse=True)
        
        # Start with the most important memory
        compressed = [sorted_by_importance[0]]
        considered = {0}
        
        # Consider remaining memories in order of importance
        for i in range(1, len(sorted_by_importance)):
            if len(compressed) >= target_count:
                break
                
            memory = sorted_by_importance[i]
            
            # Check if this memory is similar to any already compressed
            is_similar = False
            for existing in compressed:
                # Skip if states aren't tensors or have different shapes
                if not isinstance(memory.state, torch.Tensor) or not isinstance(existing.state, torch.Tensor):
                    continue
                    
                if memory.state.shape != existing.state.shape:
                    continue
                
                # Calculate similarity
                dot_product = torch.sum(memory.state * existing.state).item()
                norm_a = torch.norm(memory.state).item()
                norm_b = torch.norm(existing.state).item()
                
                if norm_a == 0 or norm_b == 0:
                    similarity = 0
                else:
                    similarity = dot_product / (norm_a * norm_b)
                
                if similarity > self.similarity_threshold:
                    is_similar = True
                    break
            
            # Add to compressed if not similar to existing memories
            if not is_similar:
                compressed.append(memory)
                considered.add(i)
        
        return compressed
