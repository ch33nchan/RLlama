#!/usr/bin/env python3
"""
Basic usage example for RLlama.
"""

import yaml
import os
import sys

# Add parent dir to path to run directly
sys.path.append(os.path.abspath(".."))

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
    reward = engine.compute(context)
    print(f"Response {i+1} (length {len(response)}): Reward = {reward}")

print("\n✅ Example completed successfully!")
