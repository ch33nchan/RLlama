# examples/bayesian_optimization_example.py

import sys
import os
import random
from typing import Dict

# Ensure RLlama is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rllama.optimization.bayesian_optimizer import BayesianRewardOptimizer

def objective_function(shaping_config: Dict) -> float:
    """
    A dummy objective function that simulates an RL training run.

    In a real scenario, this function would:
    1. Take the `shaping_config` dictionary.
    2. Use it to configure and run a full RL training loop (e.g., with TRL).
    3. Return a final performance metric, like the average return over the
       last N episodes, which we want to maximize.

    For this example, we'll simulate a simple relationship: the score is
    maximized when the LengthReward weight is close to an arbitrary "ideal" value.
    """
    print(f"\n--- Running trial with config: {shaping_config} ---")
    
    # Get the weight suggested by Optuna for this trial
    length_weight = shaping_config.get("LengthReward", {}).get("weight", 0.0)
    
    # Simulate a performance metric. Let's assume the ideal weight is ~1.2
    ideal_weight = 1.2
    
    # The score is a Gaussian curve centered at the ideal weight.
    # The further the trial weight is from the ideal, the lower the score.
    score = 100 * (1 / (1 + (length_weight - ideal_weight)**2))
    
    # Add some random noise to make it more realistic
    score += random.uniform(-2, 2)
    
    print(f"Trial complete. Simulated performance score: {score:.4f}")
    
    return score


def main():
    """Main function to run the optimization."""
    print("Running Bayesian Optimization Example for RLlama")
    
    config_path = "examples/optimizer_config.yaml"
    
    # 1. Initialize the optimizer
    try:
        optimizer = BayesianRewardOptimizer(
            config_path=config_path,
            objective_function=objective_function,
            direction="maximize"
        )
    except Exception as e:
        print(f"Error initializing optimizer: {e}")
        return

    # 2. Run the optimization process
    best_params = optimizer.optimize(n_trials=50, show_progress_bar=True)

    print("\n--- Final Results ---")
    print("The optimizer has found the best combination of weights:")
    print(best_params)

if __name__ == "__main__":
    main()