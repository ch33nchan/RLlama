#!/usr/bin/env python3
"""
Example demonstrating the memory systems in RLlama.
"""

import torch
import time
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama.memory import MemoryEntry, EpisodicMemory, WorkingMemory, MemoryCompressor

def main():
    """Run memory systems example"""
    print("RLlama Memory Systems Example")
    print("=" * 40)
    
    # Create memory systems
    episodic_memory = EpisodicMemory(capacity=100)
    working_memory = WorkingMemory(max_size=5)
    
    print("\n1. Creating and storing memories")
    print("-" * 40)
    
    # Create some test embeddings (simulate state representations)
    embeddings = [torch.randn(10) for _ in range(10)]
    
    # Add to episodic memory
    for i, emb in enumerate(embeddings):
        entry = MemoryEntry(
            state=emb,
            action=i % 3,  # Some action id
            reward=float(i) / 10,
            next_state=None,
            done=False,
            timestamp=int(time.time()) + i,
            importance=float(i) / 10
        )
        episodic_memory.add(entry)
    
    print(f"Added {len(embeddings)} memories to episodic memory")
    
    # Print some memory statistics
    print(f"Memory capacity: {episodic_memory.capacity}")
    print(f"Current memory size: {len(episodic_memory.memories)}")
    
    print("\n2. Retrieving relevant memories")
    print("-" * 40)
    
    # Create a query embedding
    query = torch.randn(10)
    print("Creating a query embedding")
    
    # Retrieve similar memories
    relevant_memories = episodic_memory.retrieve_relevant(query, k=3)
    
    print(f"Retrieved {len(relevant_memories)} most relevant memories")
    for i, memory in enumerate(relevant_memories):
        print(f"Memory {i+1}:")
        print(f"  - Action: {memory.action}")
        print(f"  - Reward: {memory.reward:.4f}")
        print(f"  - Timestamp: {memory.timestamp}")
    
    print("\n3. Working memory")
    print("-" * 40)
    
    # Add some embeddings to working memory
    for i, emb in enumerate(embeddings[:5]):
        working_memory.add(emb)
        print(f"Added embedding {i+1} to working memory")
    
    # Generate context
    context = working_memory.get_context(query)
    print(f"Generated context with shape: {context.shape}")
    
    print("\n4. Memory compression")
    print("-" * 40)
    
    # Create a memory compressor
    compressor = MemoryCompressor(compression_ratio=0.5)
    
    # Get all memories
    all_memories = episodic_memory.memories
    
    # Compress memories
    compressed = compressor.compress(all_memories)
    print(f"Compressed {len(all_memories)} memories to {len(compressed)} memories")
    
    print("\n✅ Memory systems example completed successfully!")

if __name__ == "__main__":
    main()
