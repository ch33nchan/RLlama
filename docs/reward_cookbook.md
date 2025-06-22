# RLlama Reward Shaping Cookbook (LLM RLF Focus)

This guide provides practical recipes for using the `rllama.rewards` framework to engineer sophisticated reward signals for Reinforcement Learning Fine-tuning (RLF) of Large Language Models (LLMs).

## Core Concepts Recap

*   **Reward Components (`BaseReward`)**: Define individual reward sources. For LLMs, this includes preference scores, safety penalties, instruction adherence metrics, verbosity costs, etc. Implement by inheriting `rllama.rewards.base.BaseReward`.
*   **Reward Composer (`RewardComposer`)**: Aggregates raw values from components and combines them using dynamic weights. Handles optional normalization. Found in `rllama.rewards.composition`.
*   **Reward Shaper (`RewardShaper` & `RewardConfig`)**: Manages how component weights change over training steps (e.g., curriculum learning, fading penalties). Configured via YAML/Python. Found in `rllama.rewards.shaping`.
*   **Registry (`rllama.rewards.registry`)**: Maps string names to component classes for loading from YAML.
*   **YAML Configuration**: Declaratively define the entire reward strategy for reproducibility and experimentation.

---

## LLM RLF Recipes & Techniques

### Recipe 1: Combining Preference Scores with Constraint Penalties

*   **Goal:** Fine-tune an LLM using a primary preference model score while penalizing constraint violations (e.g., toxicity, verbosity).
*   **Concept:** Use one `BaseReward` component to pass through the score from your external preference model (often available in the `info` dict during PPO). Add other `BaseReward` components for penalties (e.g., a `ToxicityPenalty` checking the generated text, a `LengthPenalty` based on token count). Combine them using the `RewardComposer`.
*   **Example (`reward_config.yaml`):**
    ```yaml
    composer_settings:
      normalize: false # Often preference scores are already somewhat scaled

    reward_components:
      preference_score:
        class: InfoValueReward # Generic component to extract a value from the info dict
        params:
          info_key: "preference_score" # Key where the score is stored
          default_value: 0.0
      toxicity_penalty:
        class: ToxicityPenalty # Assumes custom implementation registered
        params:
          penalty_value: -2.0 # Applied if toxic content detected
          toxicity_threshold: 0.9 # From a hypothetical toxicity classifier
      length_penalty:
        class: LengthPenalty # Assumes custom implementation registered
        params:
          max_length: 256
          penalty_per_token: -0.01

    reward_shaping:
      preference_score:
        initial_weight: 1.0 # Use preference score directly
        decay_schedule: 'none'
      toxicity_penalty:
        initial_weight: 1.0 # Apply penalty fully
        decay_schedule: 'none'
      length_penalty:
        initial_weight: 0.5 # Start with a moderate length penalty
        decay_schedule: 'linear' # Gradually reduce penalty weight if needed
        decay_steps: 10000
        min_weight: 0.1
    ```
*   **Implementation Notes:** The custom penalty components (`ToxicityPenalty`, `LengthPenalty`) need access to the generated `action` (response text/tokens) and potentially other `info` data within their `__call__` methods.

---

### Recipe 2: Curriculum Learning for Instruction Complexity

*   **Goal:** Train an LLM to follow increasingly complex instructions by adjusting reward weights over time.
*   **Concept:** Start with high weight on rewards for following simple instructions (e.g., correct format) and low weight for complex reasoning rewards. Gradually decrease the simple instruction weight and increase the complex reasoning weight using the `RewardShaper`.
*   **Example (`reward_config.yaml`):**
    ```yaml
    # ... composer_settings ...
    reward_components:
      formatting_check:
        class: FormattingReward # Checks if output matches required JSON/Markdown etc.
        params:
          reward_value: 1.0
      reasoning_quality:
        class: ReasoningScoreReward # Extracts score based on complex task success
        params:
          info_key: "reasoning_eval_score"
          default_value: 0.0

    reward_shaping:
      formatting_check:
        initial_weight: 5.0 # Emphasize basic formatting early
        decay_schedule: 'linear'
        decay_steps: 20000
        min_weight: 0.5
      reasoning_quality:
        initial_weight: 1.0 # Start lower weight for complex task
        decay_schedule: 'linear_increase' # Custom schedule needed, or use 'none' and start high later
        decay_steps: 20000 # Increase over time
        max_weight: 5.0 # Cap the weight
        start_step: 5000 # Start increasing after 5k steps (requires adding start_step logic)
    ```
*   **Implementation Notes:** Requires implementing the specific reward components and potentially custom decay schedules in `rllama.rewards.shaping`.

---

### Recipe 3: Normalizing Diverse Reward Signals

*   **Goal:** Combine signals with vastly different scales (e.g., preference score [-5, 5], toxicity penalty {0, -10}, length bonus [0, 1]) without one dominating.
*   **Concept:** Specify a `normalization_strategy` (e.g., `'z_score'`, `'min_max'`) in the `composer_settings` section of your YAML config. The `RewardComposer` will then attempt to normalize the raw component values before applying the weights defined in `reward_shaping`. *(Note: Normalization implementation is currently pending, setting a strategy will log a warning)*. If set to `None` or omitted, no normalization occurs.
*   **Example (`reward_config.yaml`):**
    ```yaml
    composer_settings:
      normalization_strategy: 'z_score' # Options: 'z_score', 'min_max', null (or omit for None)
      # norm_window: 1000 # Parameters like window/epsilon would be needed for actual implementation
      # norm_epsilon: 1e-8

    reward_components:
      # ... (preference_score, toxicity_penalty, etc.) ...

    reward_shaping:
      # Weights now apply to normalized signals (once implemented), making them more comparable
      preference_score:
        initial_weight: 1.0
        decay_schedule: 'none'
      toxicity_penalty:
        initial_weight: 1.5 # Weight relative to preference score after normalization
        decay_schedule: 'none'
      # ... other components ...
    ```
*   **Considerations:** Normalization helps balance signals but introduces non-stationarity. Implementation details like window size (`norm_window`) and epsilon (`norm_epsilon`) would be important. Requires a warm-up period for statistics to stabilize.

---

### Recipe 4: Potential-Based Shaping for Smooth Guidance (Advanced)

*   **Goal:** Provide dense guidance towards a desired state (e.g., specific response characteristics) without altering the optimal policy defined by the primary reward (like the preference score).
*   **Concept:** Define a potential function `Phi(state)` based on response properties (e.g., closeness to target length, embedding similarity to an ideal response). The shaping reward `F` is `gamma * Phi(next_state) - Phi(state)`. Add this `F` to the main reward signal.
*   **Example (`reward_config.yaml`):**
    ```yaml
    reward_components:
      base_reward:
        class: InfoValueReward # The main preference score or env reward
        params:
          info_key: "preference_score"
      pbrs_guidance:
        class: LLMPotentialBasedReward # Custom PBRS implementation for LLMs
        params:
          potential_function: "embedding_similarity" # Identifier for Phi logic
          gamma: 0.99 # Agent's discount factor (from PPO config)
          potential_params:
            target_embedding: [ ... ] # Embedding of an ideal response

    reward_shaping:
      base_reward:
        initial_weight: 1.0
        decay_schedule: 'none'
      pbrs_guidance:
        initial_weight: 0.1 # PBRS is often scaled lower
        decay_schedule: 'none'
    ```
*   **Implementation Notes:** Requires a sophisticated `LLMPotentialBasedReward` component that computes `Phi` based on the LLM's output (`action`/`next_state`). Ensure the main reward (`base_reward`) is also included.

---

### Recipe 5: Optimizing Reward Hyperparameters with Bayesian Optimization

*   **Goal:** Automatically find the best `initial_weight`, `decay_rate`, `normalization_strategy`, etc., for your reward components to maximize LLM alignment metrics (e.g., win rate against reference model, evaluation score on a benchmark).
*   **Concept:** Use `rllama.rewards.optimization.BayesianRewardOptimizer` with Optuna. Define a search space over the parameters in your `reward_config.yaml` and an `objective` function that runs a PPO fine-tuning trial and returns the alignment metric.
*   **Prerequisites:** Install Optuna (`pip install optuna`).
*   **Example (`my_opt_config.yaml` - Base):**
    ```yaml
    # Base config - parameters here might be overridden by Optuna
    composer_settings:
      normalization_strategy: null # Optuna can choose this
      # norm_window: 1000 # Could also be tuned if implemented
    reward_components:
      # ... (preference_score, toxicity_penalty, etc.) ...
    reward_shaping:
      preference_score:
        initial_weight: 1.0
      toxicity_penalty:
        initial_weight: 1.0
      # ...
    ```
*   **Example (`run_optimization.py` - Snippets):**
    ```python
    # --- Define Search Space ---
    search_space = {
        "composer_settings": { # Can tune composer settings too
            "normalization_strategy": {"type": "categorical", "choices": [None, "z_score", "min_max"]} # Tune strategy
            # "norm_window": {"type": "int", "low": 500, "high": 5000} # If implemented
        },
        "reward_shaping": { # Target shaping parameters
            "preference_score": {
                "initial_weight": {"type": "float", "low": 0.5, "high": 2.0}
            },
            "toxicity_penalty": {
                "initial_weight": {"type": "float", "low": 0.1, "high": 5.0, "log": True}
            },
            # ... other components/parameters to tune ...
        }
    }

    # --- Implement Objective Function ---
    # def objective(trial: optuna.Trial, base_config: Dict, search_space: Dict) -> float:
    #     # 1. Get suggested config from trial (optimizer handles this)
    #     # 2. Setup PPO Trainer (TRL) and RLlama Composer/Shaper with suggested config
    #     # 3. Run PPO fine-tuning for a fixed number of steps/epochs
    #     # 4. Evaluate the fine-tuned model (e.g., win rate, benchmark score)
    #     # 5. Return the evaluation metric (higher is better for 'maximize')
    #     # Use template: examples/optimizer_template.py

    # --- Run Optimizer ---
    # optimizer = BayesianRewardOptimizer(...)
    # study = optimizer.optimize()
    # print(f"Best trial: {study.best_trial.value}")
    # print(f"Best params: {study.best_params}")
    ```
*   **Execution & Analysis:** Run the script. Analyze Optuna results (best parameters, parameter importance) using logs or `optuna-dashboard`.

---

*Refer to the main [README.md](../../README.md) for core component details and integration examples.*