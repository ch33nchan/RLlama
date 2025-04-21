import torch
import numpy as np
from rllama.rewards.shaping import RewardShaper, RewardConfig
from rllama.rewards.composition import RewardComposer
from rllama.rewards.visualization import RewardDashboard # Assuming this exists
from rllama.rewards.common import GoalReward, StepPenaltyReward, ActionNoveltyReward # Import common rewards

# --- Initialize Components ---

# 1. Define Reward Components (using BaseReward classes)
reward_components = [
    GoalReward(goal_key="goal_reached", reward_value=10.0), # Use a larger reward for goal
    StepPenaltyReward(penalty=-0.05), # Slightly larger penalty
    ActionNoveltyReward(frequency_key="action_frequencies")
]
composer = RewardComposer(reward_components)

# 2. Configure Reward Shaping
reward_configs = {
    "goal_reward": RewardConfig(name="goal_reward", initial_weight=1.0, decay_schedule='none'),
    "step_penalty": RewardConfig(name="step_penalty", initial_weight=1.0, decay_schedule='none'),
    "action_novelty": RewardConfig(name="action_novelty", initial_weight=0.5, decay_schedule='linear', min_weight=0.01, decay_steps=500) # Use decay_steps
}
# Ensure configs match component names
assert all(comp.name in reward_configs for comp in reward_components), "Mismatch between reward components and configs"
shaper = RewardShaper(reward_configs)

# 3. (Optional) Setup Dashboard
dashboard = RewardDashboard() # Assuming this exists

# --- Helper Function for Reward Calculation (User Convenience) ---
def calculate_reward(state, action, next_state, info, composer, shaper, global_step):
    """Calculates the final shaped reward for a transition."""
    # Update shaper state (e.g., for decay)
    shaper.update_weights(global_step=global_step) # Pass global step
    current_weights = shaper.get_weights()

    # Compute raw rewards
    raw_rewards = composer.compute_rewards(state, action, next_state, info)

    # Combine rewards
    final_reward = composer.combine_rewards(raw_rewards, current_weights)

    return final_reward, raw_rewards, current_weights

# --- Simulate Training Loop ---
MAX_STEPS = 1000
action_frequencies = {} # Simple tracking for novelty demo

for step in range(MAX_STEPS):
    # 1. Simulate environment step (get state, action, next_state, info)
    # --- Dummy data ---
    current_state = {"pos": np.random.rand(2)}
    action = np.random.randint(0, 4)
    next_state = {"pos": np.random.rand(2)}
    info = {
        "goal_reached": (step % 100 == 99),
        "action_frequencies": action_frequencies
    }
    # --- End Dummy data ---

    # 2. Calculate Shaped Reward (using the helper function)
    final_reward, raw_rewards, current_weights = calculate_reward(
        current_state, action, next_state, info, composer, shaper, step
    )

    # 3. Use final_reward for RL update (e.g., store in buffer)
    # (RL algorithm update logic goes here, using final_reward)

    # 4. Log data
    dashboard.log_iteration(weights=current_weights, metrics=raw_rewards, step=step)

    # --- Update dummy info for next step ---
    try:
        action_key = hash(action)
    except TypeError:
        action_key = hash(str(action))
    action_frequencies[action_key] = action_frequencies.get(action_key, 0) + 1
    # ---

    if step % 100 == 0 or step == MAX_STEPS - 1:
        print(f"--- Step {step} ---")
        print(f"Raw Rewards: {raw_rewards}")
        print(f"Current Weights: { {k: f'{v:.3f}' for k, v in current_weights.items()} }") # Formatted weights
        print(f"Final Reward: {final_reward:.4f}")

# 5. Generate Dashboard (after training)
# dashboard.generate_dashboard("reward_integration_analysis.html")

print("\nDemo finished.")