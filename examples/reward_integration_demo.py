import gymnasium as gym
import numpy as np
import yaml # Import yaml
import sys # For error handling
import os # To construct absolute path for config

# --- Framework Imports ---
from rllama.rewards.shaping import RewardShaper, RewardConfig
from rllama.rewards.composition import RewardComposer
from rllama.rewards.visualization import RewardDashboard
from rllama.rewards.base import BaseReward
from rllama.rewards.registry import create_reward_component, register_reward_component # Import registry functions

from typing import Any, Dict, List

# --- Define FrozenLake Specific Rewards (Keep here for demo purposes) ---
# Alternatively, move to common.py and add to registry if generally useful

class FrozenLakeGoalReward(BaseReward):
    """Reward reaching the goal state (G)."""
    @property
    def name(self) -> str:
        return "frozen_lake_goal" # This name MUST match the key in reward_shaping config

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        terminated = info.get("terminated", False)
        is_goal_state = (next_state == 15) # Assuming 4x4 map
        return 10.0 if terminated and is_goal_state else 0.0

class FrozenLakeHolePenalty(BaseReward):
    """Penalize falling into a hole (H)."""
    @property
    def name(self) -> str:
        return "frozen_lake_hole" # This name MUST match the key in reward_shaping config

    def __call__(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        terminated = info.get("terminated", False)
        is_goal_state = (next_state == 15) # Assuming 4x4 map
        return -5.0 if terminated and not is_goal_state else 0.0

# --- Register Demo-Specific Components ---
# This allows the registry to find them even if defined locally
register_reward_component("frozen_lake_goal", FrozenLakeGoalReward)
register_reward_component("frozen_lake_hole", FrozenLakeHolePenalty)
# StepPenalty is already in the main registry

# --- Load Configuration ---
config_path = os.path.join(os.path.dirname(__file__), "reward_config.yaml") # Path relative to this script
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from: {config_path}")
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading or parsing configuration: {e}")
    sys.exit(1)

# --- Initialize Components based on Config ---

# 1. Create Reward Components using Registry
reward_components_config = config.get('reward_components', {})
reward_components: List[BaseReward] = []
component_names_from_config = []
for name, comp_config in reward_components_config.items():
    class_name = comp_config.get('class', name) # Use key as default class name if not specified
    params = comp_config.get('params', {})
    try:
        component = create_reward_component(class_name, **params)
        # Important: Check if the instantiated component's name matches expectations if needed
        # We rely on the reward_shaping keys matching the component's .name property later
        reward_components.append(component)
        component_names_from_config.append(component.name) # Store the actual .name property
    except (ValueError, TypeError) as e:
        print(f"Error creating reward component '{name}' (class: {class_name}): {e}")
        sys.exit(1)

print(f"Instantiated reward components: {[comp.name for comp in reward_components]}")


# 2. Initialize Reward Composer with optional normalization
composer_settings = config.get('composer_settings', {})
normalize = composer_settings.get('normalize', False)
norm_window = composer_settings.get('norm_window', 1000)
composer = RewardComposer(reward_components, normalize=normalize, norm_window=norm_window)
print(f"RewardComposer initialized (Normalization: {normalize}, Window: {norm_window})")


# 3. Configure Reward Shaping from Config
shaping_configs_dict = config.get('reward_shaping', {})
# Convert loaded dict to RewardConfig objects
try:
    # Ensure every key in shaping_configs_dict has a 'name' field or add it
    for name, cfg_dict in shaping_configs_dict.items():
        cfg_dict['name'] = name # Ensure name is part of the dict for RewardConfig init
    reward_configs = {name: RewardConfig(**cfg_dict) for name, cfg_dict in shaping_configs_dict.items()}
except TypeError as e:
     print(f"Error creating RewardConfig objects from config: {e}")
     print("Ensure all necessary fields (initial_weight, etc.) are present in reward_shaping config.")
     sys.exit(1)

# Validate that all instantiated components have a shaping config
component_names_actual = [comp.name for comp in reward_components]
missing_shaping_configs = set(component_names_actual) - set(reward_configs.keys())
if missing_shaping_configs:
    print(f"Error: Missing reward_shaping configuration for components: {missing_shaping_configs}")
    sys.exit(1)
# Validate that all shaping configs correspond to an instantiated component
extra_shaping_configs = set(reward_configs.keys()) - set(component_names_actual)
if extra_shaping_configs:
     print(f"Warning: Extra reward_shaping configurations found for non-instantiated components: {extra_shaping_configs}")
     # Optionally remove them or just proceed
     for extra_key in extra_shaping_configs:
         del reward_configs[extra_key]


shaper = RewardShaper(reward_configs)
print(f"RewardShaper initialized with configs for: {list(reward_configs.keys())}")


# 4. Setup Dashboard
dashboard = RewardDashboard()

# --- Helper Function (remains the same) ---
def calculate_reward(state, action, next_state, info, composer, shaper, global_step):
    """Calculates the final shaped reward for a transition."""
    shaper.update_weights(global_step=global_step)
    current_weights = shaper.get_weights()
    raw_rewards = composer.compute_rewards(state, action, next_state, info)
    final_reward = composer.combine_rewards(raw_rewards, current_weights)
    return final_reward, raw_rewards, current_weights

# --- Add a Simple Q-Learning Agent ---

class QLearningAgent:
    def __init__(self, observation_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon_start=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.q_table = np.zeros((observation_space.n, action_space.n))
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample() # Explore
        else:
            return np.argmax(self.q_table[state, :]) # Exploit

    def learn(self, state, action, reward, next_state, done):
        best_next_action_value = np.max(self.q_table[next_state, :])
        current_value = self.q_table[state, action]
        # Q-learning update rule
        new_value = current_value + self.lr * (reward + self.gamma * best_next_action_value * (1 - done) - current_value)
        self.q_table[state, action] = new_value

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon(self):
        return self.epsilon

# --- End Q-Learning Agent ---


# --- Initialize Components based on Config ---

# --- Simulate Training Loop with Gymnasium ---
MAX_EPISODES = 1000 # Increase episodes for learning
MAX_STEPS_PER_EPISODE = 100

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human") # Keep render mode

# --- Initialize Agent ---
agent = QLearningAgent(env.observation_space, env.action_space, epsilon_decay=0.9995) # Adjust decay if needed

global_step = 0
print(f"Action Space: {env.action_space}")
print(f"Observation Space: {env.observation_space}")
print(f"Using Q-Learning Agent")

episode_rewards_history = [] # Track episode rewards for simple analysis

for episode in range(MAX_EPISODES):
    state, info = env.reset()
    episode_reward = 0
    episode_steps = 0

    for step in range(MAX_STEPS_PER_EPISODE):
        # --- Agent Logic (Use Q-Learning Agent) ---
        action = agent.choose_action(state)
        # --- End Agent Logic ---

        next_state, reward_original, terminated, truncated, info_env = env.step(action)
        done = terminated or truncated
        info_env['terminated'] = terminated
        info_env['truncated'] = truncated

        # --- Calculate Shaped Reward ---
        final_reward, raw_rewards, current_weights = calculate_reward(
            state, action, next_state, info_env, composer, shaper, global_step
        )
        # --- End Calculate Shaped Reward ---

        # --- RL Update (Agent learns using the final_reward) ---
        agent.learn(state, action, final_reward, next_state, done)
        # --- End RL Update ---

        # --- Logging ---
        dashboard.log_iteration(weights=current_weights, metrics=raw_rewards, step=global_step)
        episode_reward += final_reward
        # --- End Logging ---

        state = next_state
        global_step += 1
        episode_steps += 1

        if done:
            break

    episode_rewards_history.append(episode_reward) # Log total episode reward

    if episode % 50 == 0 or episode == MAX_EPISODES - 1:
         print(f"--- Episode {episode} ---")
         print(f"Steps: {episode_steps}, Total Steps: {global_step}, Epsilon: {agent.get_epsilon():.3f}")
         if 'raw_rewards' in locals() and 'current_weights' in locals():
             print(f"Raw Rewards (Last Step): {raw_rewards}")
             print(f"Current Weights (Last Step): { {k: f'{v:.3f}' for k, v in current_weights.items()} }")
             print(f"Final Shaped Reward (Last Step): {final_reward:.4f}")
         print(f"Total Shaped Reward (Episode): {episode_reward:.4f}")
         # Print simple moving average of rewards
         if len(episode_rewards_history) >= 50:
             avg_reward = np.mean(episode_rewards_history[-50:])
             print(f"Avg Reward (Last 50 Episodes): {avg_reward:.4f}")


# 5. Generate Dashboard (after training)
output_file = "frozen_lake_reward_dashboard.html" # Changed name slightly
dashboard.generate_dashboard(output_file) # Now this should generate the HTML file

env.close()
print(f"\nDemo finished.")
print(f"To view dashboard: open {output_file}")

# Optional: Basic plot of episode rewards
try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(episode_rewards_history)
    # Calculate and plot moving average
    moving_avg = np.convolve(episode_rewards_history, np.ones(50)/50, mode='valid')
    plt.plot(np.arange(len(moving_avg)) + 49, moving_avg, label='50-episode avg') # Offset x-axis for moving average
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Shaped Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("episode_rewards_plot.png")
    print("Saved episode rewards plot to episode_rewards_plot.png")
    # plt.show() # Uncomment to display plot immediately
except ImportError:
    print("\nInstall matplotlib (pip install matplotlib) to see the episode rewards plot.")