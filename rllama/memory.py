import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import deque
import time

@dataclass
class MemoryEntry:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    timestamp: int
    importance: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if 'creation_time' not in self.metadata:
            self.metadata['creation_time'] = time.time()

class EpisodicMemory:
    def __init__(self, capacity: int = 10000, importance_decay: float = 0.99):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.importance_threshold = 0.5
        self.importance_decay = importance_decay
        
    def add(self, entry: MemoryEntry):
        # Update importance based on reward and timestamp
        entry.importance = abs(entry.reward) * (self.importance_decay ** (time.time() - entry.metadata['creation_time']))
        self.memories.append(entry)
        
    def retrieve_relevant(self, current_state, k: int = 5, threshold: float = None) -> List[MemoryEntry]:
        if not self.memories:
            return []
        
        # Calculate similarities and importance scores
        similarities = [self._compute_similarity(current_state, m.state) for m in self.memories]
        importance_scores = [m.importance for m in self.memories]
        
        # Combine similarity and importance
        combined_scores = [0.7 * sim + 0.3 * imp for sim, imp in zip(similarities, importance_scores)]
        
        if threshold is not None:
            filtered_indices = [i for i, score in enumerate(combined_scores) if score > threshold]
            return [self.memories[i] for i in filtered_indices[:k]]
        
        indices = np.argsort(combined_scores)[-k:]
        return [self.memories[i] for i in indices]

    def _compute_similarity(self, state1, state2) -> float:
        if isinstance(state1, torch.Tensor) and isinstance(state2, torch.Tensor):
            return torch.cosine_similarity(state1.flatten(), state2.flatten(), dim=0).item()
        return 0.0

class WorkingMemory:
    def __init__(self, max_size: int = 10, attention_temperature: float = 1.0):
        self.memory = deque(maxlen=max_size)
        self.attention_weights = None
        self.attention_temperature = attention_temperature
        
    def add(self, item: Any, importance: float = None):
        if isinstance(item, MemoryEntry):
            self.memory.append((item, importance or item.importance))
        else:
            self.memory.append((item, importance or 1.0))
        
    def get_context(self, query: torch.Tensor) -> torch.Tensor:
        if not self.memory:
            return query
            
        items, importances = zip(*self.memory)
        memory_tensor = torch.stack([m for m in items if isinstance(m, torch.Tensor)])
        importance_tensor = torch.tensor(importances).to(query.device)
        
        self.attention_weights = self._compute_attention(query, memory_tensor, importance_tensor)
        context = torch.sum(memory_tensor * self.attention_weights.unsqueeze(-1), dim=0)
        return context
        
    def _compute_attention(self, query: torch.Tensor, keys: torch.Tensor, importances: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1)
        scores = scores * importances  # Weight by importance
        return torch.softmax(scores / self.attention_temperature, dim=0)

class MemoryCompressor:
    def __init__(self, compression_ratio: float = 0.5, strategy: str = 'importance'):
        self.compression_ratio = compression_ratio
        self.strategy = strategy
        
    def compress(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        if not memories:
            return []
            
        n_keep = int(len(memories) * self.compression_ratio)
        
        if self.strategy == 'importance':
            scores = [m.importance for m in memories]
        elif self.strategy == 'recency':
            scores = [-m.timestamp for m in memories]  # Negative to sort descending
        elif self.strategy == 'hybrid':
            scores = [m.importance * (0.99 ** (time.time() - m.metadata['creation_time'])) 
                     for m in memories]
        else:
            raise ValueError(f"Unknown compression strategy: {self.strategy}")
            
        indices = np.argsort(scores)[-n_keep:]
        return [memories[i] for i in indices]