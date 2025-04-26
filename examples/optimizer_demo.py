import gymnasium as gym
import numpy as np
import yaml
import optuna
import os
import logging
from collections import deque
from typing import Dict, Any

# Ensure rllama is importable (adjust path if necessary or install with 'pip install -e .')
try:
    # print("Attempting to import RewardComposer...") # Removed debug print
    from rllama.rewards import RewardComposer
    # print("OK: RewardComposer") # Removed debug print

    # print("Attempting to import RewardShaper...") # Removed debug print
    from rllama.rewards import RewardShaper
    # print("OK: RewardShaper") # Removed debug print

    # print("Attempting to import RewardConfig...") # Removed debug print
    from rllama.rewards import RewardConfig
    # print("OK: RewardConfig") # Removed debug print

    # print("Attempting to import get_reward_component...") # Removed debug print
    from rllama.rewards import get_reward_component
    # print("OK: get_reward_component") # Removed debug print

    # print("Attempting to import BayesianRewardOptimizer...") # Removed debug print
    from rllama.rewards import BayesianRewardOptimizer
    # print("OK: BayesianRewardOptimizer") # Removed debug print

    # We don't need the dashboard for this specific demo

    # The registry should auto-register common components upon import
    # print("Attempting to import _registry_instance...") # Removed debug print
    from rllama.rewards.registry import _registry_instance
    # print("OK: _registry_instance") # Removed debug print

except ImportError as e:
    # print(f"ERROR during import: {e}") # Keep or remove this enhanced error message as you prefer
    print("Error importing RLlama components. Make sure RLlama is installed ('pip install -e .') or in the Python path.")
    print(e) # Restored original error print
    exit(1)
except Exception as e: # Catch other potential errors during import
    print(f"UNEXPECTED ERROR during import: {e}")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Simple Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, done):
        best_next_action_value = np.max(self.q_table[next_state, :]) if not done else 0
        current_q = self.q_table[state, action]
        new_q = current_q + self.lr * (reward + self.gamma * best_next_action_value - current_q)
        self.q_table[state, action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial, base_config: Dict[str, Any], search_space: Dict[str, Any]) -> float:
    """
    Runs a short RL training session with the sampled config and returns performance.
    """
    logger.info(f"\n--- Starting Trial {trial.number} ---")

    # 1. Create the specific config for this trial
    current_config = base_config.copy() # Start with base
    current_shaping = current_config.setdefault("reward_shaping", {})

    # Sample hyperparameters defined in search_space
    for comp_name, params in search_space.items():
        if comp_name not in current_shaping:
             current_shaping[comp_name] = {} # Ensure component exists in shaping section

        for param_name, definition in params.items():
            suggest_type = definition["type"]
            optuna_param_name = f"{comp_name}_{param_name}"

            if suggest_type == "float":
                value = trial.suggest_float(optuna_param_name, definition["low"], definition["high"], log=definition.get("log", False))
            elif suggest_type == "int":
                value = trial.suggest_int(optuna_param_name, definition["low"], definition["high"], log=definition.get("log", False))
            elif suggest_type == "categorical":
                value = trial.suggest_categorical(optuna_param_name, definition["choices"])
            else:
                raise ValueError(f"Unsupported type {suggest_type}")
            current_shaping[comp_name][param_name] = value

    # Ensure essential defaults if not tuned (e.g., decay_schedule)
    for comp_name, shaping_params in current_shaping.items():
        shaping_params.setdefault('decay_schedule', 'none')
        # Initial weight should be sampled or come from base, handled by Optuna wrapper now

    logger.info("Sampled Config for Trial:")
    logger.info(yaml.dump({'reward_shaping': current_shaping}))


    # 2. Set up RL environment and agent
    try:
        # Use 'is_slippery=False' for easier debugging/faster convergence in demo
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    except Exception as e:
        logger.error(f"Failed to create Gym environment: {e}")
        # Return a very bad score if env creation fails
        return -float('inf')

    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = QLearningAgent(state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1) # Lower epsilon for faster exploitation

    # 3. Set up RLlama reward system using the trial's config
    try:
        # Create components
        reward_components_config = current_config.get('reward_components', {})
        reward_components = []
        component_map = {} # Keep track for shaper config
        for name, comp_config in reward_components_config.items():
            component = get_reward_component(comp_config['class'], **comp_config.get('params', {}))
            reward_components.append(component)
            component_map[component.name] = component # Map internal name to instance

        # Create Composer
        composer_settings = current_config.get('composer_settings', {})
        composer = RewardComposer(reward_components, **composer_settings)

        # Create Shaper
        shaping_configs_dict = current_config.get('reward_shaping', {})
        reward_configs = {}
        for comp_internal_name, cfg_dict in shaping_configs_dict.items():
             # Ensure the component exists before creating config
             if comp_internal_name in component_map:
                 # The key of shaping_configs_dict IS the name we need
                 cfg_dict['name'] = comp_internal_name
                 reward_configs[comp_internal_name] = RewardConfig(**cfg_dict)
             else:
                 logger.warning(f"Shaping config found for '{comp_internal_name}', but no matching component was created. Skipping.")

        # Add default configs (weight=1, schedule=none) for components that exist but have no shaping config
        for comp_name, component in component_map.items():
            if comp_name not in reward_configs:
                logger.debug(f"Component '{comp_name}' has no explicit shaping config. Adding default (weight=1, schedule=none).")
                reward_configs[comp_name] = RewardConfig(name=comp_name, initial_weight=1.0, decay_schedule='none')


        shaper = RewardShaper(reward_configs)

    except Exception as e:
        logger.error(f"Trial {trial.number}: Failed to set up RLlama components: {e}", exc_info=True)
        return -float('inf') # Return bad score

    # 4. Run short training loop
    num_episodes = 150 # Short run for demonstration
    max_steps_per_episode = 100
    episode_rewards = deque(maxlen=50) # Track rewards of last 50 episodes for evaluation

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- RLlama Integration ---
            # Update shaper (though weights are constant in this simple setup)
            # Pass global_step=episode*max_steps_per_episode + steps if using time-varying weights
            shaper.update_weights(global_step=0) # Step doesn't matter if schedule is 'none'
            current_weights = shaper.get_weights()

            # Add necessary info for reward components
            # For FrozenLake, let's explicitly add goal/hole info based on termination
            map_desc = getattr(env.unwrapped, 'desc', None) # Get map if possible
            info['env_desc'] = map_desc
            info['goal_state'] = 15 # Assuming 4x4 map goal
            info['terminated'] = terminated
            info['truncated'] = truncated
            info['reached_goal'] = terminated and next_state == info['goal_state']
            # HolePenaltyReward needs a way to know it landed on a hole
            # We can infer this if terminated but not at goal (specific to FrozenLake)
            info['landed_on_hole'] = terminated and not info['reached_goal']


            # Calculate raw and combined rewards
            raw_rewards = composer.compute_rewards(state, action, next_state, info)
            final_shaped_reward = composer.combine_rewards(raw_rewards, current_weights)
            # --- End RLlama Integration ---

            agent.learn(state, action, final_shaped_reward, next_state, done)

            state = next_state
            total_reward += final_shaped_reward # Use shaped reward for learning AND evaluation metric
            steps += 1

        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
             logger.debug(f"Trial {trial.number} - Episode {episode+1}/{num_episodes} finished. Avg Reward (last 50): {np.mean(episode_rewards):.2f}")


    # 5. Return performance metric (average reward of last N episodes)
    final_performance = np.mean(episode_rewards) if episode_rewards else -float('inf')
    logger.info(f"--- Finished Trial {trial.number} | Final Avg Reward: {final_performance:.3f} ---")
    env.close()
    return final_performance


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Base Configuration
    config_path = "/Users/cheencheen/Desktop/git/rl/RLlama/examples/optimizer_demo_config.yaml"
    try:
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        logger.info(f"Loaded base configuration from: {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        exit(1)

    # 2. Define Search Space for Optuna
    # This defines WHICH parameters Optuna should tune and their ranges/choices.
    # Keys should match the component names in reward_shaping in the YAML.
    # Parameter names within each component dict should match RewardConfig fields (e.g., 'initial_weight').
    search_space = {
        "goal_reward": { # Matches the key in reward_shaping
            "initial_weight": {"type": "float", "low": 0.1, "high": 20.0, "log": True}
        },
        "step_penalty": { # Matches the key in reward_shaping
            "initial_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True}
        },
        "hole_penalty": { # Matches the key in reward_shaping
            "initial_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True}
        }
        # Add other parameters to tune here if needed (e.g., decay parameters if using schedules)
    }
    logger.info("Defined Optuna search space:")
    logger.info(yaml.dump({'search_space': search_space}))


    # 3. Create BayesianRewardOptimizer instance (Uncomment this block)
    # Use a lambda to pass base_config and search_space to the objective wrapper inside the optimizer
    objective_with_context = lambda trial: objective(trial, base_config, search_space)

    # Define study storage path
    storage_dir = "/Users/cheencheen/Desktop/git/rl/RLlama/examples/optuna_results"
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(storage_dir, 'reward_opt_demo.db')}"
    study_name = "frozenlake_reward_tuning_demo"

    logger.info(f"Setting up BayesianRewardOptimizer. Study: '{study_name}', Storage: '{storage_path}'")

    optimizer = BayesianRewardOptimizer( # Uncomment this instantiation
        base_config=base_config,
        search_space=search_space, # Pass the search space here for validation and structure
        objective_fn=objective_with_context, # Pass the lambda wrapper
        n_trials=20,  # Number of optimization trials (increase for real run)
        study_name=study_name,
        storage=storage_path,
        direction="maximize" # We want to maximize the average episode reward
    )

    # 4. Run Optimization (Uncomment this block)
    logger.info(f"Starting optimization with {optimizer.n_trials} trials...") # Uncomment this line
    try:
        study = optimizer.optimize() # Uncomment this line

        logger.info("\n--- Optimization Finished ---")
        logger.info(f"Study name: {study.study_name}")
        logger.info(f"Number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best value (Avg Reward): {best_trial.value:.4f}")
        logger.info("Best parameters found:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")

        # You can add Optuna visualization calls here if desired
        # e.g., optuna.visualization.plot_optimization_history(study).show()
        # optuna.visualization.plot_param_importances(study).show()

    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True) # Uncomment this block

    # --- Temporary code to just run one trial directly --- (Comment out or remove this block)
    # logger.info("\n--- Running a single test trial ---")
    # # Create a dummy trial object for testing the objective function directly
    # dummy_study = optuna.create_study(direction="maximize")
    # dummy_trial = optuna.trial.Trial(dummy_study, dummy_study.ask())
    # # Manually set parameters for the dummy trial based on base_config's reward_shaping
    # test_params = {}
    # base_shaping = base_config.get("reward_shaping", {})
    # for comp_name, params_def in search_space.items():
    #     for param_name, _ in params_def.items():
    #         optuna_param_name = f"{comp_name}_{param_name}"
    #         # Get default from base_config if available, else use a fallback (e.g., midpoint)
    #         default_value = base_shaping.get(comp_name, {}).get(param_name, 1.0) # Fallback to 1.0
    #         test_params[optuna_param_name] = default_value
    #             # Note: This manual setting bypasses Optuna's suggestion logic for the test run
    #             # It uses the defaults from the YAML's reward_shaping section.

    # # Inject the manually set parameters into the dummy trial
    # dummy_study.add_trial(
    #     optuna.trial.create_trial(
    #         params=test_params,
    #         distributions={ # Need to provide distributions matching search space for Optuna internals
    #             f"{cn}_{pn}": optuna.distributions.FloatDistribution(pd["low"], pd["high"], log=pd.get("log", False))
    #             if pd["type"] == "float" else optuna.distributions.IntDistribution(pd["low"], pd["high"], log=pd.get("log", False))
    #             if pd["type"] == "int" else optuna.distributions.CategoricalDistribution(pd["choices"])
    #             for cn, ps in search_space.items() for pn, pd in ps.items()
    #         },
    #         value=None # Value will be set by the objective function
    #     )
    # )
    # # Get the actual trial object added to the study
    # actual_dummy_trial = dummy_study.trials[-1]


    # final_avg_reward = objective(actual_dummy_trial, base_config, search_space)
    # logger.info(f"--- Finished single test trial | Final Avg Reward: {final_avg_reward:.3f} ---")
    # logger.info(f"Best trial parameters (based on single run):")
    # for key, value in actual_dummy_trial.params.items():
    #      logger.info(f"  {key}: {value}")
    # logger.info(f"Best trial value: {final_avg_reward:.3f}")