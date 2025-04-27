# ... (existing content) ...

## Recipe: Optimizing Reward Shaping with Bayesian Optimization

Finding the right weights and decay schedules for reward components can be time-consuming. RLlama's `BayesianRewardOptimizer` leverages Optuna to automate this process.

**Goal:** Automatically tune reward shaping parameters (like `initial_weight`) to maximize agent performance.

**Steps:**

1.  **Prerequisites:**
    *   Install Optuna: `pip install optuna`
    *   (Optional, for visualization) Install Plotly: `pip install plotly`
    *   Have your custom reward components defined and registered.

2.  **Create Configuration (`my_opt_config.yaml`):**
    Define the components and the base shaping parameters you want to tune.

    ```yaml
    # my_opt_config.yaml
    composer_settings:
      normalize: false

    reward_components:
      my_goal_reward:
        class: MyRegisteredGoalReward # Your component class name
        params: { goal_value: 100 }
      my_step_penalty:
        class: MyRegisteredStepPenalty
        params: { penalty: -0.1 }

    reward_shaping:
      # We want Optuna to tune these weights
      MyRegisteredGoalReward: # Match component's .name property
        initial_weight: 10.0 # Default, Optuna overrides
        decay_schedule: 'none'
      MyRegisteredStepPenalty: # Match component's .name property
        initial_weight: 1.0 # Default, Optuna overrides
        decay_schedule: 'none'
    ```

3.  **Create Optimization Script (`run_optimization.py`):**
    Use `examples/optimizer_template.py` as a starting point and modify it:

    ```python
    # run_optimization.py (Simplified structure based on template)
    import gymnasium as gym
    import numpy as np
    import yaml
    import optuna
    import os
    import logging
    from rllama.rewards import BayesianRewardOptimizer, RewardComposer, RewardShaper, RewardConfig, get_reward_component
    # from my_components import MyRegisteredGoalReward, MyRegisteredStepPenalty # Import yours

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # <<< Define Your Agent >>>
    class YourAgent: # ... (Implement your agent) ...
        def __init__(self, state_size, action_size): pass
        def select_action(self, state): return 0 # Placeholder
        def learn(self, s, a, r, ns, d, i): pass # Placeholder

    # <<< Implement Your Objective Function >>>
    def objective(trial: optuna.Trial, base_config: Dict, search_space: Dict) -> float:
        logger.info(f"--- Trial {trial.number} ---")
        current_config = base_config.copy() # Optimizer wrapper handles sampling
        current_shaping = current_config.setdefault("reward_shaping", {})
        # Populate current_shaping based on trial.params (as in template)
        # ... (Logic as in optimizer_template.py objective function) ...

        # --- Setup Env, Agent, RLlama (as in template) ---
        try:
            # env = gym.make("YourEnv-vX") # Your environment
            env = gym.make("CartPole-v1") # Example
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = YourAgent(state_size, action_size)

            # Setup RLlama Composer/Shaper using current_config
            # ... (Logic as in optimizer_template.py objective function) ...
            reward_components_config = current_config.get('reward_components', {})
            # ... create components, composer, shaper ...
            composer = RewardComposer(...) # Placeholder
            shaper = RewardShaper(...) # Placeholder

        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            return -float('inf')

        # --- Run RL Loop (as in template) ---
        total_rewards = []
        num_episodes = 50 # Adjust as needed
        for episode in range(num_episodes):
            # ... (Your RL episode loop using agent, env, composer, shaper) ...
            # state, info = env.reset()
            # episode_reward = 0
            # done = False
            # step = 0
            # while not done:
            #    action = agent.select_action(state)
            #    next_state, base_reward, term, trunc, info = env.step(action)
            #    done = term or trunc
            #    shaped_reward = composer.calculate_reward(...)
            #    shaped_reward = shaper.shape_reward(shaped_reward, step)
            #    agent.learn(state, action, shaped_reward, next_state, done, info)
            #    state = next_state
            #    episode_reward += base_reward # Or shaped_reward
            #    step += 1
            # total_rewards.append(episode_reward)
            total_rewards.append(np.random.rand() * 10) # Placeholder reward

        env.close()

        # --- Return Performance Metric ---
        metric = np.mean(total_rewards[-10:]) # Example: Avg reward last 10 episodes
        logger.info(f"--- Trial {trial.number} Finished | Metric: {metric:.3f} ---")
        return metric

    # --- Main Execution ---
    if __name__ == "__main__":
        config_path = "my_opt_config.yaml" # Your config file
        with open(config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        # Define Search Space
        search_space = {
            "MyRegisteredGoalReward": { # Match component name
                "initial_weight": {"type": "float", "low": 1.0, "high": 50.0, "log": True}
            },
            "MyRegisteredStepPenalty": { # Match component name
                "initial_weight": {"type": "float", "low": 0.01, "high": 2.0, "log": True}
            }
        }

        objective_with_context = lambda trial: objective(trial, base_config, search_space)

        storage_path = "sqlite:///my_opt_study.db"
        study_name = "my_env_reward_tuning"

        optimizer = BayesianRewardOptimizer(
            base_config=base_config,
            search_space=search_space,
            objective_fn=objective_with_context,
            n_trials=30, # Number of trials
            study_name=study_name,
            storage=storage_path,
            direction="maximize" # Or "minimize"
        )

        study = optimizer.optimize()

        # (Optional) Add visualization and best config saving here
        # ...
    ```

4.  **Run the Optimization:**
    ```bash
    python run_optimization.py
    ```
    Optuna will run multiple trials, calling your `objective` function with different reward shaping parameters each time. It will print the progress and report the best parameters found.

5.  **Analyze Results:**
    Use the printed output, the saved `best_config.yaml` (if added), or Optuna visualization tools (like `optuna-dashboard sqlite:///my_opt_study.db`) to understand the results.

This recipe provides a framework. You'll need to fill in the details specific to your RL environment, agent, and reward