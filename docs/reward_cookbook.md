# RLlama Reward Shaping Cookbook

This guide explains how to use the `rllama.rewards` framework to design, combine, and dynamically shape reward signals for your Reinforcement Learning agents. It provides recipes for common reward shaping patterns.

## Core Concepts Recap

*   **Reward Components (`BaseReward`)**: Building blocks representing single reward sources (goal, penalty, etc.). Inherit from `rllama.rewards.base.BaseReward`.
*   **Reward Composer (`RewardComposer`)**: Calculates raw values from components and combines them using weights. Found in `rllama.rewards.composition`.
*   **Reward Shaper (`RewardShaper` & `RewardConfig`)**: Manages dynamic weights based on schedules. Found in `rllama.rewards.shaping`.
*   **Registry (`rllama.rewards.registry`)**: Maps string names to component classes for easy loading from configuration.
*   **YAML Configuration**: Define components, composer settings, and shaping schedules declaratively.

---

## Recipes & Techniques

Here are some common patterns and advanced techniques you can implement using the framework:

### Recipe 1: Curriculum Learning via Weight Scheduling

*   **Goal:** Train an agent on a complex task by initially emphasizing simpler sub-goals and gradually increasing the weight of the main task reward.
*   **Concept:** Use different decay/increase schedules for reward components. Start with a high weight for an "easy" reward (e.g., reaching an intermediate checkpoint) and a low weight for the "hard" final goal reward. Schedule the easy reward weight to decay while the hard reward weight increases or stays high.
*   **Example (`reward_config.yaml`):**
    ```yaml
    reward_components:
      checkpoint:
        class: CheckpointReward # Assumes custom reward registered
        params:
          target_region: [10, 10, 5, 5]
      final_goal:
        class: GoalReward # Assumes standard goal reward registered
        params:
          goal_key: "is_success"
      step_penalty:
        class: step_penalty # Common component, likely pre-registered
        params:
          penalty: -0.01

    reward_shaping:
      checkpoint_reward: # Name matches CheckpointReward().name
        initial_weight: 5.0
        decay_schedule: 'linear'
        decay_steps: 50000 # Decay over 50k steps
        min_weight: 0.1
      goal_reward: # Name matches GoalReward().name
        initial_weight: 10.0 # Keep final goal weight high
        decay_schedule: 'none'
      step_penalty: # Matches StepPenaltyReward().name
        initial_weight: 1.0
        decay_schedule: 'none'
    ```
*   **Implementation Notes:** You might need to implement custom decay schedules (e.g., `linear_increase`) or add a `start_step` parameter to `RewardConfig` and `RewardShaper` for more complex curricula.

---

### Recipe 2: Combining Intrinsic Curiosity with Extrinsic Rewards

*   **Goal:** Encourage exploration in sparse reward environments using an intrinsic motivation signal (like novelty) alongside the main task reward.
*   **Concept:** Implement a reward component that measures state novelty (e.g., using a count-based approach, or prediction error from a learned model). Combine this intrinsic reward with the extrinsic task reward. The intrinsic reward's weight might be decayed over time.
*   **Example (`reward_config.yaml`):**
    ```yaml
    reward_components:
      novelty:
        class: StateNoveltyReward # Assumes custom implementation registered
        params:
          buffer_size: 10000
          novelty_scale: 0.1
      goal:
        class: GoalReward
        params:
          goal_key: "is_success"

    reward_shaping:
      state_novelty: # Matches StateNoveltyReward().name
        initial_weight: 2.0
        decay_schedule: 'exponential'
        decay_rate: 0.9999 # Decay slowly
        decay_steps: 1 # Decay happens per step for exponential
        min_weight: 0.05
      goal_reward: # Matches GoalReward().name
        initial_weight: 10.0
        decay_schedule: 'none'
    ```
*   **Implementation Notes:** The `StateNoveltyReward` component needs internal logic to track visited states or manage a predictive model.

---

### Recipe 3: Normalizing Rewards with Different Scales

*   **Goal:** Combine reward components with very different scales (e.g., dense distance [-1, 0] vs. sparse goal +100) without unintended dominance.
*   **Concept:** Enable normalization in the `RewardComposer`. It calculates running statistics (mean, std dev) for each raw reward component and scales them before applying weights.
*   **Example (`reward_config.yaml`):**
    ```yaml
    composer_settings:
      normalize: true # Enable normalization
      norm_window: 5000 # Calculate stats over the last 5000 steps
      norm_epsilon: 1e-8 # Prevent division by zero

    reward_components:
      distance:
        class: DistanceToGoalReward # Assumes custom implementation registered
        params:
          scale: -1.0
      goal:
        class: GoalReward
        params:
          goal_key: "is_success"
          reward_value: 100.0

    reward_shaping:
      distance_reward: # Matches DistanceToGoalReward().name
        initial_weight: 1.0 # Weights now apply to normalized values
        decay_schedule: 'none'
      goal_reward: # Matches GoalReward().name
        initial_weight: 1.0 # Weights now apply to normalized values
        decay_schedule: 'none'
    ```
*   **Considerations:** Normalization introduces non-stationarity. The `norm_window` is a key hyperparameter. Requires a warm-up period.

---

### Recipe 4: Potential-Based Shaping (PBRS)

*   **Goal:** Add dense shaping rewards without changing the optimal policy of the underlying MDP.
*   **Concept:** Define a potential function `Phi(s)`. The shaping reward `F` is `gamma * Phi(s') - Phi(s)`. This guarantees policy invariance. The final reward is `R_env + F`.
*   **Example (`reward_config.yaml`):**
    ```yaml
    reward_components:
      environment:
        class: EnvironmentReward # Component passing env_reward from info dict
        params:
          reward_key: "env_reward"
      pbrs:
        class: PotentialBasedReward # Assumes custom implementation registered
        params:
          potential_function: "inverse_distance" # Identifier for potential logic
          gamma: 0.99 # Agent's discount factor
          potential_params:
            target_pos: [10, 10]

    reward_shaping:
      environment_reward: # Matches EnvironmentReward().name
        initial_weight: 1.0 # Original reward weight is 1
        decay_schedule: 'none'
      potential_shaping: # Matches PotentialBasedReward().name
        initial_weight: 1.0 # PBRS component weight is 1
        decay_schedule: 'none'
    ```
*   **Implementation Notes:** `PotentialBasedReward` needs `gamma` and logic to compute `Phi(s)`. Ensure the original environment reward is also included.

---

### Recipe 5: Action Penalty

*   **Goal:** Discourage specific actions or high-magnitude continuous actions.
*   **Concept:** Create a component returning a negative value based on the `action`.
*   **Example (`reward_config.yaml`):**
    ```yaml
    reward_components:
      goal:
        class: GoalReward
        params:
          goal_key: "is_success"
      action_cost:
        class: ActionMagnitudePenalty # Assumes custom implementation registered
        params:
          penalty_scale: 0.001

    reward_shaping:
      goal_reward:
        initial_weight: 1.0
        decay_schedule: 'none'
      action_penalty: # Matches ActionMagnitudePenalty().name
        initial_weight: 1.0
        decay_schedule: 'none'
    ```
*   **Implementation Notes:** Component's `__call__` uses the `action` argument (e.g., `-penalty_scale * ||action||^2`).

---

### Recipe 6: Survival Bonus

*   **Goal:** Encourage longevity when the main goal is sparse.
*   **Concept:** Provide a small, constant positive reward for each non-terminal step.
*   **Example (`reward_config.yaml`):**
    ```yaml
    reward_components:
      goal:
        class: GoalReward
        params:
          goal_key: "is_success"
          reward_value: 100.0
      survival:
        class: SurvivalBonusReward # Assumes custom implementation registered
        params:
          bonus_per_step: 0.01

    reward_shaping:
      goal_reward:
        initial_weight: 1.0
        decay_schedule: 'none'
      survival_bonus: # Matches SurvivalBonusReward().name
        initial_weight: 1.0
        decay_schedule: 'none'
    ```
*   **Implementation Notes:** Component's `__call__` returns `bonus_per_step`.

---

*Find more examples and integration details in the main [README.md](../../README.md) and the example scripts.*