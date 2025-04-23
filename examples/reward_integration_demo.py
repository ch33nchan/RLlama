import gymnasium as gym # Import gymnasium
import numpy as np
# Removed torch import as it's not used in this basic demo
import yaml # Add yaml import
from rllama.rewards.shaping import RewardShaper, RewardConfig
from rllama.rewards.composition import RewardComposer
from rllama.rewards.visualization import RewardDashboard # Assuming this exists
# Let's define specific rewards for FrozenLake
from rllama.rewards.base import BaseReward
from typing import Any, Dict

# --- Define FrozenLake Specific Rewards ---

class FrozenLakeGoalReward(BaseReward):
    """Reward reaching the goal state (G)."""
    @property
    def name(self) -> str:
        return "frozen_lake_goal"

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        # The default FrozenLake reward is 1 for goal, 0 otherwise
        # We can access the original reward via info if needed, or just check termination
        # Let's use the environment's termination signal and check if it was the goal state (state 15 in 4x4)
        terminated = info.get("terminated", False)
        is_goal_state = (next_state == 15) # Assuming 4x4 map
        return 10.0 if terminated and is_goal_state else 0.0

class FrozenLakeHolePenalty(BaseReward):
    """Penalize falling into a hole (H)."""
    @property
    def name(self) -> str:
        return "frozen_lake_hole"

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        terminated = info.get("terminated", False)
        is_goal_state = (next_state == 15) # Assuming 4x4 map
        # Penalty if terminated but NOT at the goal
        return -5.0 if terminated and not is_goal_state else 0.0

class StepPenaltyReward(BaseReward): # Reusing the concept
    """Applies a constant penalty for each step taken."""
    @property
    def name(self) -> str:
        return "step_penalty"

    def __init__(self, penalty: float = -0.01):
        self._penalty = penalty

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        return self._penalty

# --- Initialize Components ---

# 1. Define Reward Components (remains the same)
reward_components = [
    FrozenLakeGoalReward(),
    FrozenLakeHolePenalty(),
    StepPenaltyReward(penalty=-0.01),
]
composer = RewardComposer(reward_components)

# 2. Load Reward Shaping Configuration from YAML
config_path = "examples/reward_config.yaml" # Path relative to project root
try:
    with open(config_path, 'r') as f:
        loaded_configs_dict = yaml.safe_load(f)
    # Convert loaded dict to RewardConfig objects
    reward_configs = {name: RewardConfig(**cfg) for name, cfg in loaded_configs_dict.items()}
    print(f"Loaded reward configurations from: {config_path}")
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_path}")
    # Define default configs or exit
    reward_configs = { # Fallback example
         "frozen_lake_goal": RewardConfig(name="frozen_lake_goal", initial_weight=1.0),
         "frozen_lake_hole": RewardConfig(name="frozen_lake_hole", initial_weight=1.0),
         "step_penalty": RewardConfig(name="step_penalty", initial_weight=1.0)
    }
    print("Warning: Using default reward configurations.")
except Exception as e:
    print(f"Error loading or parsing reward config: {e}")
    # Handle error appropriately
    exit()


# Ensure configs match component names (check after loading)
assert all(comp.name in reward_configs for comp in reward_components), \
    f"Mismatch between reward components ({[c.name for c in reward_components]}) and loaded configs ({list(reward_configs.keys())})"
shaper = RewardShaper(reward_configs)

# 3. Setup Dashboard (remains the same)
dashboard = RewardDashboard()

# --- Helper Function (remains the same) ---
def calculate_reward(state, action, next_state, info, composer, shaper, global_step):
    """Calculates the final shaped reward for a transition."""
    shaper.update_weights(global_step=global_step)
    current_weights = shaper.get_weights()
    raw_rewards = composer.compute_rewards(state, action, next_state, info)
    final_reward = composer.combine_rewards(raw_rewards, current_weights)
    return final_reward, raw_rewards, current_weights

# --- Simulate Training Loop with Gymnasium ---
MAX_EPISODES = 500 # Run for a fixed number of episodes
MAX_STEPS_PER_EPISODE = 100 # Prevent infinite loops in one episode

# Use the is_slippery=False version for deterministic transitions, easier debugging
# env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None) # Add render_mode="human" to watch
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human") # Change None to "human"

global_step = 0
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")

for episode in range(MAX_EPISODES):
    state, info = env.reset() # Get initial state
    episode_reward = 0
    episode_steps = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # --- Agent Logic (Random Actions for Demo) ---
        action = env.action_space.sample()
        # --- End Agent Logic ---

        # --- Environment Step ---
        next_state, reward_original, terminated, truncated, info_env = env.step(action)
        # Combine termination conditions
        done = terminated or truncated
        # Add terminated/truncated flags to info for reward functions
        info_env['terminated'] = terminated
        info_env['truncated'] = truncated
        # --- End Environment Step ---

        # --- Calculate Shaped Reward ---
        final_reward, raw_rewards, current_weights = calculate_reward(
            state, action, next_state, info_env, composer, shaper, global_step
        )
        # --- End Calculate Shaped Reward ---

        # --- RL Update (Use final_reward here) ---
        # (A real agent would use final_reward to update its policy/value function)
        # --- End RL Update ---

        # --- Logging ---
        dashboard.log_iteration(weights=current_weights, metrics=raw_rewards, step=global_step)
        episode_reward += final_reward # Track shaped reward per episode
        # --- End Logging ---

        state = next_state
        global_step += 1
        episode_steps += 1

        if done:
            break # End episode

    if episode % 50 == 0 or episode == MAX_EPISODES - 1: # Print progress periodically
         print(f"--- Episode {episode} ---")
         print(f"Steps: {episode_steps}, Total Steps: {global_step}")
         print(f"Raw Rewards (Last Step): {raw_rewards}")
         print(f"Current Weights (Last Step): { {k: f'{v:.3f}' for k, v in current_weights.items()} }")
         print(f"Final Shaped Reward (Last Step): {final_reward:.4f}")
         print(f"Total Shaped Reward (Episode): {episode_reward:.4f}")


# 5. Generate Dashboard (after training)
output_file = "frozen_lake_reward_analysis.html"
dashboard.generate_dashboard(output_file) # Uncomment if dashboard is functional
print(f"\nDemo finished. Dashboard data logged.")
# print(f"To view dashboard (if generated): open {output_file}")

env.close()