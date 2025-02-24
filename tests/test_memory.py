import sys
import os
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rllama.memory import MemoryEntry, EpisodicMemory, WorkingMemory, MemoryCompressor

def test_memory_components():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create some test data with proper dimensions
    state = torch.randn(1, 768).to(device)
    action = "test_action"
    reward = 1.0
    next_state = torch.randn(1, 768).to(device)
    timestamp = int(datetime.now().timestamp())

    try:
        # Test MemoryEntry
        print("\nTesting MemoryEntry...")
        entry = MemoryEntry(state, action, reward, next_state, False, timestamp)
        print(f"Entry created with importance: {entry.importance}")
        print(f"Entry metadata: {entry.metadata}")

        # Test EpisodicMemory
        print("\nTesting EpisodicMemory...")
        episodic_memory = EpisodicMemory(capacity=5, importance_decay=0.99)
        
        # Add multiple entries
        for i in range(3):
            new_entry = MemoryEntry(
                torch.randn(1, 768).to(device),
                f"action_{i}",
                float(i),
                torch.randn(1, 768).to(device),
                False,
                timestamp + i
            )
            episodic_memory.add(new_entry)
        
        # Test retrieval
        relevant_memories = episodic_memory.retrieve_relevant(state, k=2)
        print(f"Retrieved {len(relevant_memories)} relevant memories")
        print(f"First memory importance: {relevant_memories[0].importance if relevant_memories else 'None'}")

        # Test WorkingMemory
        print("\nTesting WorkingMemory...")
        working_memory = WorkingMemory(max_size=3, attention_temperature=0.5)
        
        # Add items with different importances
        for i in range(3):
            working_memory.add(torch.randn(768).to(device), importance=float(i))
        
        # Test context generation
        query = torch.randn(768).to(device)
        context = working_memory.get_context(query)
        print(f"Context shape: {context.shape}")
        print(f"Attention weights: {working_memory.attention_weights}")

        # Test MemoryCompressor
        print("\nTesting MemoryCompressor...")
        compressor = MemoryCompressor(compression_ratio=0.5, strategy='hybrid')
        
        # Create test memories for compression
        test_memories = [
            MemoryEntry(
                torch.randn(1, 768).to(device),
                f"action_{i}",
                float(i),
                torch.randn(1, 768).to(device),
                False,
                timestamp + i
            )
            for i in range(4)
        ]
        
        compressed_memories = compressor.compress(test_memories)
        print(f"Compressed from {len(test_memories)} to {len(compressed_memories)} memories")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_memory_components()