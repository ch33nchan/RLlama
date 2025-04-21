import torch
import numpy as np
from rllama.rewards.shaping import RewardShaper, RewardConfig
from rllama.rewards.optimization import BayesianRewardOptimizer
from rllama.rewards.visualization import RewardDashboard

# Initialize components
reward_configs = {
    "task_completion": RewardConfig(
        name="task_completion",
        initial_weight=1.0,
        decay_schedule="linear",
        warmup_steps=1000
    ),
    "efficiency": RewardConfig(
        name="efficiency",
        initial_weight=0.5,
        decay_schedule="exponential",
        min_weight=0.1
    ),
    "diversity": RewardConfig(
        name="diversity",
        initial_weight=0.3,
        decay_schedule="linear",
        warmup_steps=500
    )
}

shaper = RewardShaper(reward_configs)
dashboard = RewardDashboard()

# Simulate some training iterations
for step in range(100):
    # Simulate some metrics
    metrics = {
        "task_completion": np.random.uniform(0.3, 0.9),
        "efficiency": np.random.uniform(0.4, 0.8),
        "diversity": np.random.uniform(0.2, 0.7)
    }
    
    # Update reward weights
    shaper.update_weights(metrics)
    
    # Log to dashboard
    dashboard.log_iteration(
        weights=shaper.current_weights,
        metrics=metrics,
        step=step
    )
    
    print(f"Step {step}:")
    print("Current weights:", shaper.current_weights)
    print("Metrics:", metrics)
    print("---")

# Generate visualization
dashboard.generate_dashboard("reward_analysis.html")