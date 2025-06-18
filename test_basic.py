#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import numpy as np

# Test core functionality
from rllama.engine import RewardEngine
from rllama.rewards.optimizer import BayesianRewardOptimizer
from rllama.memory import EpisodicMemory, WorkingMemory, MemoryEntry

def test_reward_engine():
    """Test the reward engine"""
    print("\n=== Testing Reward Engine ===")
    
    # Create a simple config file
    config = {
        "reward_components": [
            {
                "name": "LengthReward",
                "params": {
                    "target_length": 100,
                    "strength": 0.001
                }
            },
            {
                "name": "ConstantReward",
                "params": {
                    "value": 0.1
                }
            }
        ],
        "shaping_config": {
            "LengthReward": {"weight": 0.5},
            "ConstantReward": {"weight": 1.0}
        },
        "logging": {
            "log_dir": "./test_output",
            "log_frequency": 1
        }
    }

    # Write config to a temp file
    with open("./test_output/test_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    try:
        # Initialize the reward engine
        engine = RewardEngine("./test_output/test_config.yaml")
        
        # Test with different responses
        contexts = [
            {"response": "This is a short response.", "step": 0},
            {"response": "This is a much longer response that should be closer to our target length. It contains more words and should get a better reward score from the length component.", "step": 1},
            {"response": "This is an extremely long response that is well beyond our target length. It contains many, many words and should be penalized for being too verbose. Sometimes less is more, and conciseness is valued in communication. This response just keeps going with unnecessary details and filler text to make it longer than needed.", "step": 2}
        ]
        
        # Calculate rewards
        rewards = []
        for i, context in enumerate(contexts):
            reward = engine.compute_and_log(context)
            rewards.append(reward)
            print(f"Context {i+1}: Length={len(context['response'])}, Reward={reward}")
        
        print("✅ Reward engine test completed")
        return True
    except Exception as e:
        print(f"❌ Reward engine test failed with error: {e}")
        return False

def test_memory_systems():
    """Test memory systems"""
    print("\n=== Testing Memory Systems ===")
    
    try:
        # Create memory systems
        episodic_memory = EpisodicMemory(capacity=100)
        working_memory = WorkingMemory(max_size=5)
        
        # Create some test embeddings (simulate state representations)
        print("Creating test embeddings...")
        embeddings = [torch.randn(10) for _ in range(10)]
        
        # Add to episodic memory
        print("Testing episodic memory...")
        for i, emb in enumerate(embeddings):
            entry = MemoryEntry(
                state=emb,
                action=i % 3,  # Some action id
                reward=float(i) / 10,
                next_state=None,
                done=False,
                timestamp=int(100 + i),
                importance=float(i) / 10
            )
            episodic_memory.add(entry)
        
        # Test retrieval
        test_query = torch.randn(10)
        relevant_memories = episodic_memory.retrieve_relevant(test_query, k=3)
        print(f"Retrieved {len(relevant_memories)} relevant memories from episodic memory")
        
        # Test working memory
        print("Testing working memory...")
        for i, emb in enumerate(embeddings[:5]):
            working_memory.add(emb)
        
        context = working_memory.get_context(test_query)
        print(f"Generated context with shape: {context.shape}")
        
        print("✅ Memory systems test passed!")
        return True
    except Exception as e:
        print(f"❌ Memory systems test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Running basic RLlama tests...")
    
    # Run tests
    engine_success = test_reward_engine()
    memory_success = test_memory_systems()
    
    # Show overall results
    print("\n=== Test Summary ===")
    print(f"Reward Engine: {'✅ Passed' if engine_success else '❌ Failed'}")
    print(f"Memory Systems: {'✅ Passed' if memory_success else '❌ Failed'}")
    
    # Overall success
    if all([engine_success, memory_success]):
        print("\n🎉 All tests passed! RLlama is functioning correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
