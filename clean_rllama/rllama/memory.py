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

# Add this to the existing memory.py file

class MemoryCompressor:
    """
    Compresses and abstracts patterns from memories to improve efficiency.
    """
    
    def __init__(self, compression_ratio: float = 0.5, clustering_threshold: float = 0.8):
        """
        Initialize the memory compressor.
        
        Args:
            compression_ratio: Target ratio of compressed to original memory
            clustering_threshold: Similarity threshold for clustering memories
        """
        self.compression_ratio = compression_ratio
        self.clustering_threshold = clustering_threshold
        
    def compress(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Compress a list of memories by finding patterns and abstracting.
        
        Args:
            memories: List of memory entries to compress
            
        Returns:
            Compressed list of memory entries
        """
        if len(memories) <= 1:
            return memories.copy()
        
        # Calculate target size after compression
        target_size = max(1, int(len(memories) * self.compression_ratio))
        
        # Extract feature vectors for clustering
        if isinstance(memories[0].state, torch.Tensor):
            # If states are already tensors, use them directly
            feature_vectors = torch.stack([m.state for m in memories])
        else:
            # Fall back to reward values and timestamps as features
            feature_vectors = torch.tensor([[m.reward, m.timestamp] for m in memories])
        
        # Apply hierarchical clustering
        clusters = self._hierarchical_cluster(feature_vectors, target_size)
        
        # Select representative memories from each cluster
        compressed_memories = []
        for cluster_indices in clusters:
            cluster_memories = [memories[i] for i in cluster_indices]
            
            if len(cluster_memories) == 1:
                # Single item cluster, just add it
                compressed_memories.append(cluster_memories[0])
            else:
                # For multi-item clusters, select the most important memory
                representative = self._select_representative(cluster_memories)
                compressed_memories.append(representative)
        
        return compressed_memories
    
    def _hierarchical_cluster(self, feature_vectors: torch.Tensor, target_size: int) -> List[List[int]]:
        """
        Apply hierarchical clustering to feature vectors.
        
        Args:
            feature_vectors: Tensor of feature vectors to cluster
            target_size: Target number of clusters
            
        Returns:
            List of clusters, where each cluster is a list of indices
        """
        # Move to CPU for clustering
        cpu_vectors = feature_vectors.cpu().numpy()
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_vectors = scaler.fit_transform(cpu_vectors)
        
        # Apply hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(
            n_clusters=target_size,
            linkage='ward',  # Minimize variance within clusters
            distance_threshold=None
        ).fit(normalized_vectors)
        
        # Group indices by cluster
        clusters = [[] for _ in range(target_size)]
        for i, label in enumerate(clustering.labels_):
            clusters[label].append(i)
        
        return clusters
    
    def _select_representative(self, memories: List[MemoryEntry]) -> MemoryEntry:
        """
        Select the most representative memory from a cluster.
        
        Args:
            memories: List of memory entries in a cluster
            
        Returns:
            The most representative memory entry
        """
        if not memories:
            raise ValueError("Empty cluster provided")
        
        # Use importance as primary criterion
        max_importance = max(m.importance for m in memories)
        candidates = [m for m in memories if m.importance == max_importance]
        
        if len(candidates) == 1:
            return candidates[0]
        
        # If tied on importance, use recency
        latest_timestamp = max(m.timestamp for m in candidates)
        return next(m for m in candidates if m.timestamp == latest_timestamp)
    
    def abstract_patterns(self, memories: List[MemoryEntry]) -> Optional[MemoryEntry]:
        """
        Create an abstract memory that captures patterns from a group of memories.
        
        Args:
            memories: List of memory entries to abstract from
            
        Returns:
            An abstract memory entry, or None if no pattern found
        """
        if len(memories) < 3:  # Need at least 3 memories to find a pattern
            return None
        
        # Check if states are tensors we can work with
        if not all(isinstance(m.state, torch.Tensor) for m in memories):
            return None
        
        # Create an average state embedding
        state_tensors = [m.state for m in memories]
        avg_state = torch.mean(torch.stack(state_tensors), dim=0)
        
        # Calculate average reward
        avg_reward = sum(m.reward for m in memories) / len(memories)
        
        # Use latest timestamp
        latest_time = max(m.timestamp for m in memories)
        
        # Create metadata with pattern information
        pattern_metadata = {
            "abstract": True,
            "abstracted_from": len(memories),
            "avg_reward": avg_reward,
            "reward_std": np.std([m.reward for m in memories]),
            "creation_time": time.time()
        }
        
        # Create the abstract memory
        abstract_memory = MemoryEntry(
            state=avg_state,
            action=memories[0].action,  # Use first action as representative
            reward=avg_reward,
            next_state=None,
            done=False,
            timestamp=latest_time,
            importance=max(m.importance for m in memories) * 1.5,  # Boost importance
            metadata=pattern_metadata
        )
        
        return abstract_memory