#!/bin/bash

echo "Creating clean repository structure..."

# Create directory structure
mkdir -p clean_rllama/rllama/{core,rewards/{components,models},utils,integration,dashboard}
mkdir -p clean_rllama/examples/{simple,advanced}
mkdir -p clean_rllama/docs/images

# Copy core files
cp LICENSE clean_rllama/
cp README.md clean_rllama/
cp pyproject.toml clean_rllama/

# Copy main modules
cp rllama/__init__.py clean_rllama/rllama/
cp rllama/engine.py clean_rllama/rllama/core/
cp rllama/agent.py clean_rllama/rllama/
cp rllama/memory.py clean_rllama/rllama/
cp rllama/logger.py clean_rllama/rllama/core/
cp rllama/dashboard.py clean_rllama/rllama/
cp rllama/run_dashboard.py clean_rllama/rllama/

# Copy dashboard files
mkdir -p clean_rllama/rllama/dashboard
cp rllama/dashboard/*.py clean_rllama/rllama/dashboard/

# Copy reward components
cp rllama/rewards/*.py clean_rllama/rllama/rewards/
cp -r rllama/rewards/components clean_rllama/rllama/rewards/

# Copy integration files
mkdir -p clean_rllama/rllama/integration
cp rllama/integration/*.py clean_rllama/rllama/integration/

# Copy utils
mkdir -p clean_rllama/rllama/utils
cp rllama/utils/*.py clean_rllama/rllama/utils/

# Create necessary __init__.py files
touch clean_rllama/rllama/core/__init__.py
touch clean_rllama/rllama/rewards/models/__init__.py
touch clean_rllama/rllama/integration/__init__.py
touch clean_rllama/rllama/utils/__init__.py
touch clean_rllama/rllama/dashboard/__init__.py

# Copy documentation
cp -r docs/*.md clean_rllama/docs/
cp -r docs/assets/* clean_rllama/docs/images/

# Create clean examples
cat > clean_rllama/examples/simple/basic_usage.py << 'EOT'
#!/usr/bin/env python3
"""
Basic usage example for RLlama.
This demonstrates how to use the RewardEngine to calculate rewards.
"""

import yaml
import os
from rllama import RewardEngine

# Create output directory
os.makedirs("./output", exist_ok=True)

# Create a simple config
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
        "log_dir": "./output",
        "log_frequency": 1
    }
}

# Write the config to a file
with open('./output/simple_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Initialize the reward engine
engine = RewardEngine('./output/simple_config.yaml')
print("✅ RewardEngine initialized successfully!")

# Example responses
responses = [
    "This is a short response.",
    "This is a medium length response that should be close to our target length of 100 characters.",
    "This is a very long response that goes well beyond our target length. It contains many unnecessary words and should be penalized for being too verbose. Sometimes less is more, and brevity is valued in communication."
]

# Calculate rewards for each response
print("\nCalculating rewards for different response lengths:")
print("-" * 60)
for i, response in enumerate(responses):
    context = {"response": response, "step": i}
    reward = engine.compute_and_log(context)
    print(f"Response {i+1} (length {len(response)}): Reward = {reward}")

print("\n✅ Example completed successfully!")
EOT

cat > clean_rllama/examples/advanced/reward_optimization.py << 'EOT'
#!/usr/bin/env python3
"""
Advanced example showing reward optimization with RLlama.
This demonstrates how to use the BayesianRewardOptimizer to find optimal reward weights.
"""

import os
import numpy as np
import yaml
from rllama import RewardEngine
from rllama.rewards.optimizer import BayesianRewardOptimizer

# Create output directory
os.makedirs("./output", exist_ok=True)

# Create a simple base config
base_config = {
    "reward_components": [
        {
            "name": "LengthReward",
            "params": {
                "target_length": 150,
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
        "log_dir": "./output",
        "log_frequency": 1
    }
}

# Write the config to a file
with open('./output/base_config.yaml', 'w') as f:
    yaml.dump(base_config, f)

# Define parameter space for optimization
param_space = {
    "LengthReward__weight": (0.1, 1.0),
    "ConstantReward__weight": (0.5, 2.0)
}

# Create synthetic data for testing
def generate_test_data(n_samples=10):
    data = []
    lengths = np.random.randint(50, 300, n_samples)
    
    for length in lengths:
        data.append({
            "response": "x" * length,
            "ideal_score": 1.0 if abs(length - 150) < 20 else max(0, 1.0 - abs(length - 150) / 150)
        })
    
    return data

# Define evaluation function
def evaluate_reward_weights(params):
    # Apply parameters to config
    temp_config = base_config.copy()
    
    for key, value in params.items():
        component, param = key.split("__")
        if component in temp_config["shaping_config"]:
            if isinstance(temp_config["shaping_config"][component], dict):
                temp_config["shaping_config"][component][param] = value
            else:
                temp_config["shaping_config"][component] = value
    
    # Create temp config file
    with open('./output/temp_config.yaml', 'w') as f:
        yaml.dump(temp_config, f)
    
    # Create engine with this config
    engine = RewardEngine('./output/temp_config.yaml')
    
    # Evaluate on synthetic data
    test_data = generate_test_data()
    mse = 0.0
    
    for sample in test_data:
        context = {"response": sample["response"]}
        reward = engine.compute(context)
        mse += (reward - sample["ideal_score"]) ** 2
    
    mse /= len(test_data)
    return -mse  # We want to maximize negative MSE (minimize MSE)

# Create optimizer
print("Creating Bayesian Reward Optimizer...")
optimizer = BayesianRewardOptimizer(
    param_space=param_space,
    eval_function=evaluate_reward_weights,
    direction="maximize",
    n_trials=10  # Small number for quick testing
)

# Run optimization
print("Running reward weight optimization...")
print("This may take a few moments...")
results = optimizer.optimize()

print("\n=== Optimization Results ===")
print(f"Best parameters: {results.best_params}")
print(f"Best value (negative MSE): {results.best_value}")

# Generate optimized config
optimized_config = optimizer.generate_config("./output/optimized_config.yaml")

print("\nOptimized config saved to ./output/optimized_config.yaml")
print("Try running with this optimized config to see improved rewards!")

# Compare original vs optimized
print("\n=== Comparing Original vs. Optimized Config ===")
original_engine = RewardEngine('./output/base_config.yaml')
optimized_engine = RewardEngine('./output/optimized_config.yaml')

test_responses = [
    "Short response.",
    "Medium length response that should be reasonably close to our target length of 150 characters. This is a good example.",
    "Very long response that vastly exceeds our target length. It contains many unnecessary words and should be penalized for being too verbose. Brevity is valued in communication but this response just keeps going with unnecessary details."
]

for i, response in enumerate(test_responses):
    context = {"response": response}
    orig_reward = original_engine.compute(context)
    opt_reward = optimized_engine.compute(context)
    
    print(f"\nResponse {i+1} (length {len(response)}):")
    print(f"  Original reward: {orig_reward:.4f}")
    print(f"  Optimized reward: {opt_reward:.4f}")

print("\n✅ Example completed successfully!")
EOT

# Create a memory example
cat > clean_rllama/examples/advanced/memory_systems.py << 'EOT'
#!/usr/bin/env python3
"""
Example demonstrating the memory systems in RLlama.
"""

import torch
import time
import numpy as np
from rllama.memory import MemoryEntry, EpisodicMemory, WorkingMemory, MemoryCompressor

def main():
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
EOT

echo "Reorganization complete! Check the clean_rllama directory for the reorganized structure."
