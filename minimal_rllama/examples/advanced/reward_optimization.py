#!/usr/bin/env python3
"""
Advanced example showing reward optimization with RLlama.
"""

import os
import numpy as np
import yaml
import sys

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

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

# Save base config
with open('./output/base_config.yaml', 'w') as f:
    yaml.dump(base_config, f)

def generate_test_data(n_samples=10):
    """Generate synthetic test data"""
    data = []
    lengths = np.random.randint(50, 300, n_samples)
    
    for length in lengths:
        data.append({
            "response": "x" * length,
            "ideal_score": 1.0 if abs(length - 150) < 20 else max(0, 1.0 - abs(length - 150) / 150)
        })
    
    return data

def evaluate_reward_weights(params):
    """Evaluate a set of reward weights"""
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

def main():
    """Run the optimization example"""
    print("Creating Bayesian Reward Optimizer...")
    
    # Define parameter space
    param_space = {
        "LengthReward__weight": (0.1, 1.0),
        "ConstantReward__weight": (0.5, 2.0)
    }
    
    # Create optimizer
    optimizer = BayesianRewardOptimizer(
        param_space=param_space,
        eval_function=evaluate_reward_weights,
        direction="maximize",
        n_trials=10  # Small number for quick testing
    )
    
    # Run optimization
    print("Running optimization...")
    results = optimizer.optimize(show_progress_bar=True)
    
    print("\n=== Optimization Results ===")
    print(f"Best parameters: {results.best_params}")
    print(f"Best value: {results.best_value}")
    
    # Generate optimized config
    optimized_config = optimizer.generate_config("./output/optimized_config.yaml")
    
    print("\nOptimized config saved to ./output/optimized_config.yaml")
    
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

if __name__ == "__main__":
    main()
