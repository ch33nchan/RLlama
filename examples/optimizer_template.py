# --- Template for Bayesian Reward Optimization ---
# Description:
# This script provides a template for using RLlama's BayesianRewardOptimizer
# to tune reward shaping parameters for your custom RL environment and agent.
#
# Instructions:
# 1. Define Your Reward Components: Create your custom reward component classes
#    inheriting from rllama.rewards.RewardComponentBase and register them.
# 2. Create Configuration YAML: Copy and modify optimizer_template_config.yaml.
#    - List your registered reward components under 'reward_components'.
#    - Define the base 'reward_shaping' parameters you want to tune.
# 3. Define Search Space: Modify the 'search_space' dictionary below to specify
#    which parameters Optuna should tune and their ranges/types.
# 4. Implement RL Logic: Fill in the `objective` function with your code to:
#    - Create your RL environment.
#    - Create your RL agent.
#    - Set up the RLlama RewardComposer and RewardShaper using the config.
#    - Run your RL training/evaluation loop using the shaped rewards.
#    - Return a performance metric (e.g., average return) to be optimized.
# 5. Run the Script: Execute this Python script.

import gymnasium as gym # Or your preferred RL environment library
import numpy as np
import yaml
import optuna
import os
import logging
from collections import deque
from typing import Dict, Any

# Ensure rllama is importable
try:
    from rllama.rewards import (
        RewardComposer,
        RewardShaper,
        RewardConfig,
        get_reward_component,
        BayesianRewardOptimizer
    )
    # Import your custom registered components if they are in separate files
    # from my_custom_rewards import MyReward1, MyReward2 # Example
except ImportError as e:
    print("Error importing RLlama components. Make sure RLlama is installed ('pip install -e .') or in the Python path.")
    print(e)
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- User Defined RL Agent (Placeholder) ---
# Replace this with your actual RL agent implementation
class YourAgent:
    def __init__(self, state_size, action_size, **kwargs):
        logger.info("Initializing YourAgent (Replace with your implementation)")
        self.state_size = state_size
        self.action_size = action_size
        # Add your agent's parameters and initialization logic

    def select_action(self, state):
        # Replace with your action selection logic
        logger.debug("Selecting action (Replace with your implementation)")
        return np.random.choice(self.action_size) # Example: random action

    def learn(self, state, action, reward, next_state, done, info):
        # Replace with your agent's learning/update logic
        logger.debug(f"Learning step (Replace with your implementation): reward={reward:.2f}, done={done}")
        pass

# --- Objective Function for Optuna ---
def objective(trial: optuna.Trial, base_config: Dict[str, Any], search_space: Dict[str, Any]) -> float:
    """
    Runs an RL training/evaluation session with the sampled config and returns performance.
    <<< THIS IS THE CORE FUNCTION YOU NEED TO IMPLEMENT >>>
    """
    logger.info(f"\n--- Starting Trial {trial.number} ---")

    # 1. Create the specific config for this trial (Handled by Optimizer Wrapper)
    #    Optuna suggests parameters based on 'search_space'.
    #    The optimizer wrapper constructs the 'current_config' for the trial.
    #    You receive the 'trial' object here to potentially use trial.report() or trial.should_prune().

    #    (Example of how config is derived, you don't need to do this manually here)
    current_config = base_config.copy()
    current_shaping = current_config.setdefault("reward_shaping", {})
    for comp_name, params in search_space.items():
        if comp_name not in current_shaping: current_shaping[comp_name] = {}
        for param_name, definition in params.items():
            optuna_param_name = f"{comp_name}_{param_name}"
            # Value is suggested by Optuna based on type in search_space
            # Example: value = trial.suggest_float(...)
            # We directly use the config generated by the optimizer wrapper,
            # but you can access the suggested params via trial.params
            if optuna_param_name in trial.params:
                 current_shaping[comp_name][param_name] = trial.params[optuna_param_name]
            else:
                 # Fallback or error if param not suggested (shouldn't happen with wrapper)
                 logger.warning(f"Parameter {optuna_param_name} not found in trial.params")

    logger.info("Sampled Config for Trial (Derived from trial.params):")
    logger.info(yaml.dump({'reward_shaping': current_shaping})) # Log the actual shaping being used

    # 2. <<< IMPLEMENT >>> Set up RL environment
    try:
        # Replace with your environment creation
        # env = gym.make('YourEnv-v0', ...)
        # Example using CartPole:
        env = gym.make('CartPole-v1')
        logger.info("Created environment: CartPole-v1 (Replace with yours)")
    except Exception as e:
        logger.error(f"Failed to create Gym environment: {e}")
        return -float('inf') # Return a very bad score

    # Get state and action space sizes (adapt as needed)
    state_size = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else env.observation_space.n
    action_size = env.action_space.n

    # 3. <<< IMPLEMENT >>> Set up RL agent
    # agent = YourAgent(state_size, action_size, ...) # Pass necessary hyperparameters
    agent = YourAgent(state_size, action_size) # Using placeholder agent

    # 4. Set up RLlama reward system using the trial's config
    try:
        # Create components listed in the config
        reward_components_config = current_config.get('reward_components', {})
        reward_components = []
        component_map = {}
        for name, comp_config in reward_components_config.items():
            # Assumes components are registered correctly
            component_class_name = comp_config['class']
            component_params = comp_config.get('params', {})
            component = get_reward_component(component_class_name, **component_params)
            reward_components.append(component)
            component_map[component.name] = component # Map internal name to instance
            logger.info(f"Initialized reward component: {component.name} (Class: {component_class_name})")

        # Create Composer
        composer_settings = current_config.get('composer_settings', {})
        composer = RewardComposer(reward_components, **composer_settings)

        # Create Shaper using the sampled parameters for this trial
        shaping_configs_dict = current_shaping # Use the sampled shaping config
        reward_configs = {}
        for comp_internal_name, cfg_dict in shaping_configs_dict.items():
             if comp_internal_name in component_map:
                 cfg_dict['name'] = comp_internal_name # Ensure name is set
                 reward_configs[comp_internal_name] = RewardConfig(**cfg_dict)
             else:
                 logger.warning(f"Shaping config found for '{comp_internal_name}', but no matching component was created. Skipping.")

        # Add default configs for components without explicit shaping
        for comp_name, component in component_map.items():
            if comp_name not in reward_configs:
                logger.debug(f"Component '{comp_name}' using default shaping (weight=1, schedule=none).")
                reward_configs[comp_name] = RewardConfig(name=comp_name, initial_weight=1.0, decay_schedule='none')

        shaper = RewardShaper(reward_configs)
        logger.info("RLlama Reward Composer and Shaper created successfully.")

    except Exception as e:
        logger.error(f"Trial {trial.number}: Failed to set up RLlama components: {e}", exc_info=True)
        return -float('inf') # Return bad score if setup fails

    # 5. <<< IMPLEMENT >>> RL Training/Evaluation Loop
    num_episodes = 100 # Example: Number of episodes to run for evaluation
    total_rewards = []
    max_steps_per_episode = 200 # Example limit

    logger.info(f"Starting RL loop for {num_episodes} episodes...")
    for episode in range(num_episodes):
        # Adapt state initialization for your environment
        state, info = env.reset()
        if isinstance(env.observation_space, gym.spaces.Discrete):
             # Q-learning style state
             pass # state is already the discrete state
        elif isinstance(env.observation_space, gym.spaces.Box):
             # DQN style state (often needs flattening or processing)
             # state = state.flatten() # Example
             pass # Keep as NumPy array for this example

        episode_reward = 0
        shaper.reset() # Reset shaper state (e.g., decay steps) for new episode

        for step in range(max_steps_per_episode):
            # a. Select action
            action = agent.select_action(state)

            # b. Step environment
            next_state, base_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Adapt state representation if needed
            if isinstance(env.observation_space, gym.spaces.Box):
                # next_state = next_state.flatten() # Example
                pass

            # c. Calculate shaped reward using RLlama
            #    Pass necessary info: state, action, next_state, base_reward, done, info
            #    Adapt the arguments based on what your components need
            shaped_reward = composer.calculate_reward(
                state=state,
                action=action,
                next_state=next_state,
                base_reward=base_reward,
                done=done,
                info=info
                # Add any other kwargs your components might need
            )
            shaped_reward = shaper.shape_reward(shaped_reward, step) # Apply weights and decay

            # d. Agent learning step
            agent.learn(state, action, shaped_reward, next_state, done, info)

            # e. Update state and episode reward
            state = next_state
            episode_reward += base_reward # Track original env reward or shaped reward? User choice.

            if done:
                break

        total_rewards.append(episode_reward)
        shaper.step_episode() # Notify shaper that episode ended (for decay)

        # Optional: Optuna Pruning - report intermediate results
        # intermediate_value = np.mean(total_rewards[-10:]) # Example: avg reward of last 10 episodes
        # trial.report(intermediate_value, episode)
        # if trial.should_prune():
        #     logger.info(f"Trial {trial.number} pruned at episode {episode}.")
        #     raise optuna.exceptions.TrialPruned()

    env.close()
    logger.info("RL loop finished.")

    # 6. <<< IMPLEMENT >>> Calculate and return performance metric
    #    Example: Average reward over the last N episodes
    if not total_rewards:
        final_metric = -float('inf') # Handle case with no rewards
    else:
        final_metric = np.mean(total_rewards[-max(1, num_episodes // 2):]) # Avg reward of last half

    logger.info(f"--- Finished Trial {trial.number} | Final Metric (Avg Reward): {final_metric:.4f} ---")
    return final_metric


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Base Configuration from YAML
    # <<< MODIFY path to your configuration file >>>
    config_path = "/Users/cheencheen/Desktop/git/rl/RLlama/examples/optimizer_template_config.yaml"
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

    # 2. <<< DEFINE >>> Search Space for Optuna
    # Define WHICH parameters Optuna should tune and their ranges/choices.
    # Keys must match component names in 'reward_shaping' in the YAML.
    # Parameter names within each component dict must match RewardConfig fields (e.g., 'initial_weight').
    search_space = {
        # Example: Tuning 'initial_weight' for two hypothetical components
        "my_positive_reward": { # Matches a key in reward_shaping
            "initial_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True}
        },
        "my_negative_penalty": { # Matches another key in reward_shaping
            "initial_weight": {"type": "float", "low": 0.01, "high": 5.0, "log": True}
        },
        # Add other components and parameters to tune here
        # Example: Tuning decay schedule (if implemented in Shaper)
        # "my_positive_reward": {
        #     "initial_weight": {"type": "float", "low": 0.1, "high": 10.0, "log": True},
        #     "decay_schedule": {"type": "categorical", "choices": ["none", "linear", "exponential"]},
        #     "decay_steps": {"type": "int", "low": 1000, "high": 10000},
        #     "decay_rate": {"type": "float", "low": 0.9, "high": 0.999}
        # }
    }
    logger.info("Defined Optuna search space:")
    logger.info(yaml.dump({'search_space': search_space}))


    # 3. Create BayesianRewardOptimizer instance
    # Use a lambda to pass base_config and search_space to the objective function
    # The objective function needs these to know the base structure and what to sample.
    objective_with_context = lambda trial: objective(trial, base_config, search_space)

    # <<< MODIFY study name and storage path >>>
    storage_dir = "/Users/cheencheen/Desktop/git/rl/RLlama/examples/optuna_results_template" # Choose a directory
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = f"sqlite:///{os.path.join(storage_dir, 'reward_opt_template.db')}"
    study_name = "my_rl_reward_tuning_template" # Choose a unique name

    logger.info(f"Setting up BayesianRewardOptimizer. Study: '{study_name}', Storage: '{storage_path}'")

    optimizer = BayesianRewardOptimizer(
        base_config=base_config,
        search_space=search_space,
        objective_fn=objective_with_context, # Pass the lambda wrapper
        n_trials=20,  # <<< SET number of optimization trials >>>
        study_name=study_name,
        storage=storage_path,
        direction="maximize" # "maximize" or "minimize" the objective's return value
    )

    # 4. Run Optimization
    logger.info(f"Starting optimization with {optimizer.n_trials} trials...")
    try:
        study = optimizer.optimize()

        logger.info("\n--- Optimization Finished ---")
        logger.info(f"Study name: {study.study_name}")
        logger.info(f"Number of finished trials: {len(study.trials)}")

        if study.best_trial:
            best_trial = study.best_trial
            logger.info(f"Best trial number: {best_trial.number}")
            logger.info(f"Best value (Metric): {best_trial.value:.4f}")
            logger.info("Best parameters found:")
            for key, value in best_trial.params.items():
                logger.info(f"  {key}: {value}")

            # --- Optional: Save the best configuration ---
            try:
                best_config = optimizer.get_best_config(study)
                best_config_path = os.path.join(storage_dir, "best_config.yaml")
                with open(best_config_path, 'w') as f:
                    yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"Saved the best configuration found to: {best_config_path}")
            except Exception as save_e:
                logger.error(f"Failed to save the best configuration: {save_e}")

            # --- Optional: Add Optuna Visualizations ---
            try:
                import optuna.visualization as vis
                if study.trials:
                    history_fig = vis.plot_optimization_history(study)
                    history_fig.show()
                    importance_fig = vis.plot_param_importances(study)
                    importance_fig.show()
                    # Add other plots like plot_slice or plot_contour if desired
            except ImportError:
                logger.warning("Optuna visualization requires plotly. Install with: pip install plotly")
            except Exception as viz_e:
                 logger.error(f"Error generating Optuna visualizations: {viz_e}")

        else:
            logger.info("No trials were completed successfully.")


    except Exception as e:
        logger.error(f"An error occurred during optimization: {e}", exc_info=True)