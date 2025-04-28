
# RLlama: Advanced Reward Engineering for LLM Fine-tuning

*A Python library providing a modular framework for designing, composing, and dynamically shaping complex reward signals for Reinforcement Learning Fine-tuning (RLF) of Large Language Models, designed to integrate seamlessly with TRL-like workflows.*

---

## The Coming Wave of RL & The Reward Engineering Bottleneck

Reinforcement Learning (RL), especially RLF for Large Language Models (LLMs), is rapidly moving from a niche technique to a foundational technology for aligning AI behavior (helpfulness, safety, instruction following). Libraries like Hugging Face TRL provide excellent interfaces for the core RL algorithms (like PPO).

However, a critical bottleneck remains: **reward engineering**. Defining the reward signal that *guides* the RL agent is often an ad-hoc, messy process:

*   **Manual Tweaking:** Developers manually combine reward sources (preference scores, penalties, bonuses) and endlessly tweak their relative weights.
*   **Static or Simple Schedules:** Changing these weights over time (reward shaping) often involves writing custom, complex loops, making curriculum learning difficult to implement systematically.
*   **Lack of Standardization:** There's no standard way to define, combine, or manage these reward components, hindering reproducibility and efficient experimentation.
*   **Optimization Challenges:** Finding the *optimal* combination and schedule of rewards is a significant hyperparameter tuning problem, often left to intuition or exhaustive manual search.

As RL becomes more widespread, this bottleneck will only intensify. We need tools to move beyond manual hacks towards **systematic reward engineering at scale.**

---

## RLlama: Your Toolkit for Scalable Reward Engineering

**RLlama is designed to be the "LangChain for RL Rewards"** – a dedicated framework that brings structure, modularity, and automation to the reward engineering process, specifically targeting LLM RLF.

Instead of writing bespoke code for combining rewards, manually adjusting weights, or implementing complex scheduling loops, RLlama provides:

*   **Modular & Reusable Reward Components (`BaseReward`):** Define individual reward sources (e.g., `PreferenceScore`, `ToxicityPenalty`, `LengthBonus`, `InstructionAdherence`) as plug-and-play Python classes. RLlama aims to provide common components out-of-the-box.
*   **Declarative Composition (`RewardComposer` & YAML):** Define *how* components are combined (summed, normalized) using simple YAML configuration, separating strategy from code.
*   **Systematic Dynamic Shaping (`RewardShaper` & YAML):** Implement sophisticated reward schedules (linear decay, exponential increase, curriculum phases) declaratively in YAML, eliminating manual loops.
*   **Automated Optimization (`BayesianRewardOptimizer`):** Leverage Bayesian optimization (via Optuna) to automatically find the most effective reward weights, shaping parameters, and normalization settings, drastically reducing manual tuning effort.
*   **Insightful Visualization (`RewardDashboard`):** Understand how each reward component contributes and how weights evolve during training, providing crucial debugging and analysis capabilities.

**Goal:** To make sophisticated reward engineering accessible and efficient, enabling developers to rapidly experiment, optimize, and deploy robust reward strategies for aligning LLMs and other RL agents, effectively acting as an **open-source toolkit for advanced RLF reward management.**

---

## Core Concepts & Components

RLlama's architecture enables this systematic approach:

1.  **`BaseReward` (`rllama.rewards.base`)**:
    *   **Purpose:** The fundamental, reusable building block for any reward source/penalty. Implement custom logic by inheriting.
    *   **LLM Context:** Define components like `PreferenceScoreReward`, `ToxicityPenalty`, `InstructionFollowingBonus`, `VerbosityPenalty`. RLlama will provide a growing library of common ones.
    *   **Implementation:** Requires `name` and `__call__(...)`. `info` dict is key for passing LLM context (query, response, external scores).
    *   **Cookbook Link:** See [Cookbook](docs/reward_cookbook.md) Recipe 1 for combining components.

2.  **`RewardComposer` (`rllama.rewards.composition`)**:
    *   **Purpose:** Aggregates `BaseReward` instances based on YAML config. Calculates raw values and combines them using current weights from the `RewardShaper`. Handles optional normalization via the `normalization_strategy` parameter (`'min_max'`, `'z_score'`, or `None`). Defaults to `None` (no normalization). *(Note: Normalization implementation is currently pending)*.
    *   **Benefit:** Decouples *what* rewards exist from *how* they are combined.
    *   **Cookbook Link:** See [Normalization Recipe (Recipe 3)](docs/reward_cookbook.md#recipe-3-normalizing-diverse-reward-signals).

3.  **`RewardShaper` & `RewardConfig` (`rllama.rewards.shaping`)**:
    *   **Purpose:** Manages dynamic component weights over time based on YAML schedules (`RewardConfig`). Calculates current weights based on `global_step`.
    *   **Benefit:** Replaces manual weight-adjustment loops with declarative configuration. Enables easy curriculum learning.
    *   **Cookbook Link:** See [Curriculum Learning (Recipe 2)](docs/reward_cookbook.md#recipe-2-curriculum-learning-for-instruction-complexity) and [Optimization (Recipe 5)](docs/reward_cookbook.md#recipe-5-optimizing-reward-hyperparameters-with-bayesian-optimization).

4.  **YAML Configuration**:
    *   **Purpose:** The central control panel. Define components, parameters, composition, normalization, and shaping schedules declaratively.
    *   **Benefit:** Enables rapid experimentation, reproducibility, and separation of concerns.

5.  **`BayesianRewardOptimizer` (`rllama.rewards.optimization`)**:
    *   **Purpose:** Automates the search for optimal reward hyperparameters (weights, schedules, etc.) using Optuna.
    *   **Benefit:** Moves beyond manual tweaking to data-driven optimization of the reward strategy.
    *   **Cookbook Link:** See [Optimization Recipe (Recipe 5)](docs/reward_cookbook.md#recipe-5-optimizing-reward-hyperparameters-with-bayesian-optimization).

6.  **`RewardDashboard` (`rllama.rewards.visualization`)**:
    *   **Purpose:** Logs and plots component values and weights over time.
    *   **Benefit:** Provides essential visibility into the complex dynamics of your engineered reward signal.

---

## Integration Workflow (Conceptual PPO Loop)

(Conceptual PPO loop code remains the same - it shows the integration points)

```python:%2FUsers%2Fcheencheen%2FDesktop%2Fgit%2Frl%2FRLlama%2Fexamples%2Fconceptual_ppo_integration.py
# --- Conceptual PPO Training Script ---
# Assume imports: TRL PPOTrainer, models, tokenizer, dataset, etc.
# Assume RLlama imports: RewardComposer, RewardShaper, RewardDashboard, load_config_from_yaml, BaseReward

# --- Define Custom RLlama Components (Example) ---
# class ToxicityPenalty(BaseReward):
#     def __init__(self, name="toxicity_penalty", penalty_value=-2.0, **kwargs): # Params from YAML
#         super().__init__(name)
#         self.penalty_value = penalty_value
#         # self.model = load_toxicity_classifier() # Load external model if needed
#     def __call__(self, state, action, next_state, info):
#         response_text = action # Assuming action is the decoded text
#         # is_toxic = self.model.predict(response_text) > 0.9
#         is_toxic = "bad word" in response_text # Simplified check
#         return self.penalty_value if is_toxic else 0.0
# # ... register this component ...

# --- Main Script ---
# 1. Load RLlama Configuration from YAML
#    This single file defines components, composition, and shaping!
reward_config_path = '/Users/cheencheen/Desktop/git/rl/RLlama/examples/configs/my_llm_reward_config.yaml'
# composer, shaper = load_config_from_yaml(reward_config_path) # Needs implementation
# dashboard = RewardDashboard()

# 2. Setup PPO Trainer (TRL)
# ppo_trainer = PPOTrainer(...)

# --- PPO Training Loop ---
# for epoch in range(num_epochs):
#     for step, batch in enumerate(ppo_trainer.dataloader):
#         global_step = ...
#
#         # 3. Generate Responses (TRL)
#         # query_tensors = batch['input_ids']
#         # response_tensors = ppo_trainer.generate(...)
#         # batch['response_text'] = tokenizer.batch_decode(response_tensors)
#         # batch['query_text'] = tokenizer.batch_decode(query_tensors)
#
#         # --- RLlama Reward Calculation ---
#         # 4. Update Reward Shaper Weights based on training progress
#         shaper.update_weights(global_step=global_step)
#         current_weights = shaper.get_weights()
#
#         # 5. Compute Rewards for the Batch
#         rewards_list = []
#         raw_rewards_log = []
#         for i in range(len(batch['query_text'])): # Process each item in the batch
#             # Prepare data for reward components
#             reward_info = {
#                 "query": batch['query_text'][i],
#                 # "preference_score": get_preference_score(batch['query_text'][i], batch['response_text'][i]), # External call
#                 # ... add any other data needed by your components ...
#             }
#             # Compute raw rewards using the composer
#             # Pass response text as 'action', other data via 'info'
#             raw_rewards_dict = composer.compute_rewards(
#                 state=None, # Often state/next_state aren't needed if info has all context
#                 action=batch['response_text'][i],
#                 next_state=None,
#                 info=reward_info
#             )
#             # Combine raw rewards using current dynamic weights
#             final_reward = composer.combine_rewards(raw_rewards_dict, current_weights)
#             rewards_list.append(torch.tensor(final_reward)) # Tensor for PPO
#             raw_rewards_log.append(raw_rewards_dict)
#
#         # Log average raw rewards for monitoring
#         # avg_raw_rewards = pd.DataFrame(raw_rewards_log).mean().to_dict()
#         # dashboard.log_iteration(weights=current_weights, metrics=avg_raw_rewards, step=global_step)
#         # --- End RLlama Integration ---
#
#         # 7. PPO Step (TRL)
#         # stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
#         # ... log stats ...

# 8. Save Dashboard Report
# dashboard.generate_dashboard(...)
```

---

## Key Components

*   **`BaseReward` (`rllama.rewards.base`):** The building block. Inherit from this to create custom reward logic (e.g., `ToxicityPenalty`, `InstructionFollowingScore`). Must implement `name` property and `__call__`. The `__call__` method receives data relevant to the LLM's generation (query, response, context, potentially base reward model scores) via its arguments (often packed into the `info` dict).
*   **`RewardComposer` (`rllama.rewards.composition`):** Aggregates `BaseReward` instances. Calculates raw values from each component (`compute_rewards`) and combines them using weights (`combine_rewards`). Supports optional normalization strategies (`'min_max'`, `'z_score'`, `None`) specified during initialization or via YAML (`composer_settings.normalization_strategy`). *(Note: Normalization implementation is currently pending)*.
*   **`RewardConfig` / `RewardShaper` (`rllama.rewards.shaping`):** Define how the weight (influence) of each `BaseReward` component changes over the training steps (e.g., linear decay, exponential increase). Configured via YAML or Python.
*   **YAML Configuration:** Centralizes the definition of components, their parameters, composition strategy (normalization), and dynamic shaping schedules. Enables easy experimentation.
*   **`BayesianRewardOptimizer` (`rllama.rewards.optimization`):** Uses Optuna to find optimal `initial_weight`, `decay_rate`, etc., in your `reward_shaping` config by running multiple training trials.
*   **`RewardDashboard` (`rllama.rewards.visualization`):** Logs and plots component values and weights over time for analysis and debugging.

---

## Examples & Templates

We provide example scripts demonstrating how to integrate RLlama into RLF workflows. These serve as starting points for your own projects:

*   **`/Users/cheencheen/Desktop/git/rl/RLlama/examples/ppo_finetuning_template/`**: *(Hypothetical)* A template showing integration with a TRL-like PPO trainer for fine-tuning an LLM (e.g., GPT-2, Llama) on a simple task. Includes:
    *   `train_ppo.py`: The main training script.
    *   `reward_config.yaml`: Example RLlama configuration.
    *   `custom_rewards.py`: Example custom `BaseReward` components.
*   **`/Users/cheencheen/Desktop/git/rl/RLlama/examples/reward_optimization_template/`**: Demonstrates using `BayesianRewardOptimizer` to tune the `reward_config.yaml` parameters.
*   **`/Users/cheencheen/Desktop/git/rl/RLlama/examples/reward_integration_demo.py`**: A simpler demo using Q-Learning and FrozenLake to illustrate core RLlama concepts (composition, shaping, dashboard) without LLM complexities.

*(Note: You will need to create/adapt these example templates based on this new focus.)*

---

## Advanced Usage & Recipes

For detailed patterns and strategies using RLlama, such as:

*   Implementing Potential-Based Reward Shaping (PBRS) for LLMs.
*   Combining preference model scores with constraint penalties.
*   Creating curriculum learning schedules via reward shaping.
*   Normalizing diverse reward signals.

See the **[Reward Shaping Cookbook](docs/reward_cookbook.md)**. *(This cookbook should also be updated to include LLM-specific recipes)*

---

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes or features before submitting a pull request. Focus areas include LLM-specific reward components, integration examples with popular RLF libraries, and performance optimizations.
