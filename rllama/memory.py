import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import deque

@dataclass
class MemoryEntry:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    timestamp: int
    importance: float = 0.0

class EpisodicMemory:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.importance_threshold = 0.5
        
    def add(self, entry: MemoryEntry):
        self.memories.append(entry)
        
    def retrieve_relevant(self, current_state, k: int = 5) -> List[MemoryEntry]:
        if not self.memories:
            return []
            
        similarities = [self._compute_similarity(current_state, m.state) for m in self.memories]
        indices = np.argsort(similarities)[-k:]
        return [self.memories[i] for i in indices]
        
    def _compute_similarity(self, state1, state2) -> float:
        if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
            return torch.cosine_similarity(state1.flatten(), state2.flatten(), dim=0).item()
        return 0.0

class WorkingMemory:
    def __init__(self, max_size: int = 10):
        self.memory = deque(maxlen=max_size)
        self.attention_weights = None
        
    def add(self, item: Any):
        self.memory.append(item)
        
    def get_context(self, query: torch.Tensor) -> torch.Tensor:
        if not self.memory:
            return query
            
        memory_tensor = torch.stack([m for m in self.memory if isinstance(m, torch.Tensor)])
        self.attention_weights = self._compute_attention(query, memory_tensor)
        context = torch.sum(memory_tensor * self.attention_weights.unsqueeze(-1), dim=0)
        return context
        
    def _compute_attention(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
        return torch.softmax(scores, dim=0)

class MemoryCompressor:
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        
    def compress(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        if not memories:
            return []
            
        n_keep = int(len(memories) * self.compression_ratio)
        importance_scores = [m.importance for m in memories]
        indices = np.argsort(importance_scores)[-n_keep:]
        return [memories[i] for i in indices]