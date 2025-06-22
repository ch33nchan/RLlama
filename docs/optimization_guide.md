
# RLlama Optimization Guide: Tuning Rewards with Bayesian Optimization

Manually tuning reward weights and scheduling parameters can be time-consuming and often leads to suboptimal results. RLlama integrates with the powerful `Optuna` library to automate this process using Bayesian Optimization via the `BayesianRewardOptimizer`.

## 1. Prerequisites: Install Optuna

If you haven't already, you need to install Optuna:

```bash
pip install optuna
```

You might also want visualization tools provided by Optuna:

```bash
pip install optuna-dashboard # For a web dashboard
pip install plotly matplotlib # For plotting functions
```

## 2. The `BayesianRewardOptimizer`

This class orchestrates the optimization process.

*   **Purpose:** To find the optimal set of reward configuration parameters (initial weights, decay rates, component-specific parameters) that maximize (or minimize) a user-defined performance metric obtained by running RL training.
*   **Core Idea:** It uses Optuna's samplers (often Tree-structured Parzen Estimator - TPE) to intelligently suggest new parameter combinations to try based on the results of previous trials.

## 3. Defining the Search Space

You need to tell the optimizer *which* parameters in your reward configuration you want to tune and *what ranges or choices* are allowed for each. This is done using a dictionary structure that mirrors your `RewardConfig` but uses Optuna's suggestion methods.

**Example `search_space` Dictionary:**

Let's assume your base `reward_config.yaml` looks like this:

```yaml
# base_config.yaml
reward_shaping:
  goal_reward:
    class: GoalReward
    params: { target_reward: 1.0 } # Parameter we might want to tune
    weight_schedule:
      initial_weight: 10.0 # Parameter we want to tune
      schedule_type: constant
  step_penalty:
    class: StepPenalty
    params: { penalty: -0.01 } # Parameter we want to tune
    weight_schedule:
      initial_weight: 1.0 # Parameter we want to tune
      schedule_type: exponential
      decay_rate: 0.9995 # Parameter we want to tune
      decay_steps: 1
      min_weight: 0.0
```

Your corresponding `search_space` dictionary in Python would look like this:

```python
import optuna

search_space = {
    "reward_shaping": {
        "goal_reward": {
            "params": {
                # Suggest a float value for target_reward between 0.5 and 5.0
                "target_reward": lambda trial: trial.suggest_float("goal_target_reward", 0.5, 5.0)
            },
            "weight_schedule": {
                # Suggest a float value for initial_weight between 1.0 and 50.0, possibly on a log scale
                "initial_weight": lambda trial: trial.suggest_float("goal_initial_weight", 1.0, 50.0, log=True)
            }
        },
        "step_penalty": {
            "params": {
                # Suggest a float value for penalty between -0.5 and -0.001
                "penalty": lambda trial: trial.suggest_float("penalty_value", -0.5, -0.001)
            },
            "weight_schedule": {
                # Suggest a float value for initial_weight between 0.1 and 10.0
                "initial_weight": lambda trial: trial.suggest_float("penalty_initial_weight", 0.1, 10.0),
                # Suggest a float value for decay_rate between 0.99 and 0.9999
                "decay_rate": lambda trial: trial.suggest_float("penalty_decay_rate", 0.99, 0.9999)
                # We could also suggest 'schedule_type' if desired:
                # "schedule_type": lambda trial: trial.suggest_categorical("penalty_schedule", ["constant", "exponential"])
            }
        }
    }
    # You could also include tunable parameters from other parts of your config,
    # like agent hyperparameters, if your objective function uses them.
    # "agent_params": {
    #    "learning_rate": lambda trial: trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # }
}
```

**Key Points:**

*   The structure mirrors your configuration file.
*   Instead of fixed values, you provide lambda functions that take an Optuna `trial` object as input.
*   Inside the lambda, use `trial.suggest_float`, `trial.suggest_int`, `trial.suggest_categorical`, etc., to define the parameter's type, range, and a unique name (e.g., `"goal_initial_weight"`). Optuna uses these names internally.

## 4. Writing the Objective Function

This is the most crucial part. The `objective` function defines *what* the optimizer is trying to achieve.

*   **Signature:** `objective(trial: optuna.trial.Trial) -> float`
*   **Purpose:**
    1.  **Receive Suggested Parameters:** Optuna calls this function, passing a `trial` object.
    2.  **Generate Config:** Use the `trial` object and your `search_space` definition to create a *complete* configuration dictionary for this specific trial. The `BayesianRewardOptimizer` often provides a helper method for this, or you can implement it manually by iterating through the `search_space` and calling the lambda functions.
    3.  **Run RL Training:** Execute a full RL training (or evaluation) run using the generated configuration. This means setting up the environment, agent, components, composer, and shaper based *on the parameters suggested for this trial*.
    4.  **Calculate Performance Metric:** Determine a single floating-point value that represents how well the agent performed with this configuration (e.g., average return over the last N episodes, success rate, minimum steps to goal).
    5.  **Return Metric:** Return this performance metric. Optuna will use this value to decide which parameters to try next.

**Conceptual `objective` Function:**

```python
import optuna
import gymnasium as gym
import numpy as np
# Assume imports for your agent, RLlama components, composer, shaper, config loading, etc.
# Assume component_registry exists (maps class names to classes)
# Assume base_config is loaded (the non-tunable parts)

def merge_configs(base, suggestions):
    """Helper to merge base config with Optuna suggestions."""
    merged = base.copy() # Start with a copy of the base
    for key, value in suggestions.items():
        if isinstance(value, dict):
            if key not in merged or not isinstance(merged[key], dict):
                merged[key] = {}
            merged[key] = merge_configs(merged.get(key, {}), value)
        else:
            # This assumes the value is the actual suggested value,
            # not the lambda function. The optimizer handles calling the lambdas.
            merged[key] = value
    return merged

def objective(trial: optuna.trial.Trial, base_config: dict, search_space: dict, component_registry: dict) -> float:
    """
    Runs an RL training loop with parameters suggested by Optuna
    and returns a performance metric.
    """
    # 1. Generate Config for this Trial
    # The BayesianRewardOptimizer might have a helper like:
    # trial_config = optimizer.generate_trial_config(trial)
    # Or manually:
    suggested_params = {}
    def apply_suggestions(space, base, target):
        for key, value in space.items():
            if isinstance(value, dict):
                target[key] = {}
                apply_suggestions(value, base.get(key, {}), target[key])
            elif callable(value): # It's a lambda function from search_space
                 # Use Optuna trial to get the actual value
                target[key] = value(trial)
            else: # Should not happen if search_space is defined correctly
                target[key] = base.get(key) # Fallback? Or error?

    apply_suggestions(search_space, base_config, suggested_params)
    trial_config = merge_configs(base_config, suggested_params) # Combine base + suggestions

    # --- Setup RL Environment and Agent based on trial_config ---
    try:
        env = gym.make("FrozenLake-v1") # Or your env
        # agent = YourAgent(**trial_config.get("agent_params", {})) # Configure agent if tuning its params

        # --- Setup RLlama Components, Composer, Shaper ---
        reward_shaping_config = trial_config.get("reward_shaping", {})
        components = {}
        for name, comp_config in reward_shaping_config.items():
            cls_name = comp_config.get("class")
            params = comp_config.get("params", {})
            if cls_name in component_registry:
                components[name] = component_registry[cls_name](**params)
            else:
                 print(f"Warning: Class {cls_name} not found.")
                 # Handle error appropriately, maybe prune trial
                 # raise optuna.TrialPruned()

        if not components:
             print("Error: No components instantiated for trial.")
             raise optuna.TrialPruned("No components") # Tell Optuna this trial failed early

        composer = RewardComposer(components)
        shaper = RewardShaper(composer, reward_shaping_config)

        # --- Run the RL Training Loop (Simplified) ---
        training_config = trial_config.get("training", {})
        num_episodes = training_config.get("num_episodes", 500) # Use fewer episodes for faster HPO
        max_steps_episode = training_config.get("max_steps_per_episode", 100)
        all_episode_rewards = []
        global_step = 0

        for episode in range(num_episodes):
            state, info = env.reset()
            terminated = truncated = False
            episode_shaped_reward = 0
            steps_in_episode = 0
            while not terminated and not truncated:
                shaper.update_weights(global_step)
                action = agent.select_action(state) # Agent uses its policy
                next_state, raw_reward, terminated, truncated, info = env.step(action)
                context = {
                    "global_step": global_step, "terminated": terminated, "truncated": truncated,
                    # Add other necessary context items
                }
                shaped_reward = shaper.shape(raw_reward, info, context)
                agent.update(state, action, shaped_reward, next_state, terminated or truncated)
                state = next_state
                episode_shaped_reward += shaped_reward
                steps_in_episode += 1
                global_step += 1
                if steps_in_episode >= max_steps_episode: truncated = True

            all_episode_rewards.append(episode_shaped_reward)

            # Optuna Pruning: Check if trial is unpromising early
            # trial.report(intermediate_value, step) # Report intermediate metric
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

        env.close()

        # --- Calculate Performance Metric ---
        # Example: Average reward of the last 10% of episodes
        if not all_episode_rewards: return -np.inf # Handle case with no rewards
        metric = np.mean(all_episode_rewards[-max(1, num_episodes // 10):])

        print(f"Trial {trial.number} finished. Metric: {metric:.4f}")
        return metric

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Consider returning a very bad value or pruning
        # return -np.inf
        raise optuna.TrialPruned(f"Exception during training: {e}")

```

## 5. Running the Optimization

Instantiate `BayesianRewardOptimizer` and call its `optimize` method.

```python
from rllama.optimization import BayesianRewardOptimizer
# Assume objective, base_config, search_space, component_registry are defined

# --- Optimizer Setup ---
optimizer = BayesianRewardOptimizer(
    base_config=base_config,
    search_space=search_space,
    objective_function=objective, # Your objective function
    objective_kwargs={ # Extra fixed arguments for your objective
        "base_config": base_config,
        "search_space": search_space,
        "component_registry": component_registry,
    },
    n_trials=100, # Number of different parameter sets to try
    study_name="rllama_reward_tuning",
    storage="sqlite:///rllama_tuning.db", # Optional: Save study results to DB
    direction="maximize" # "maximize" or "minimize" the objective metric
)

# --- Run Optimization ---
print("Starting Bayesian Optimization...")
best_params, best_value, study = optimizer.optimize()

# --- Results ---
print("\nOptimization Finished!")
print(f"Best Objective Value: {best_value}")
print("Best Parameters Found:")
import json
print(json.dumps(best_params, indent=2))

# You can now use 'best_params' to create your final, optimized reward config
```

## 6. Analyzing Results

The `optimize` method returns the Optuna `study` object. You can use Optuna's visualization functions to understand the optimization process:

```python
import optuna.visualization as vis

# Plot optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Plot parameter importance
fig = vis.plot_param_importances(study)
fig.show()

# Plot relationship between parameters and objective
fig = vis.plot_slice(study, params=["penalty_decay_rate", "goal_initial_weight"])
fig.show()

# For more visualizations and analysis, see the Optuna documentation.
# If using storage, you can use optuna-dashboard:
# optuna-dashboard sqlite:///rllama_tuning.db
```

By following these steps, you can leverage RLlama and Optuna to systematically find effective reward configurations for your RL tasks, saving significant manual effort and potentially achieving better agent performance. Refer to the `examples/optimizer_demo.py` script for a runnable implementation.
```



        