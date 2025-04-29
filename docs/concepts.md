
# RLlama Core Concepts: A Deeper Dive

This document expands on the fundamental concepts introduced in the main README, providing more detail on how RLlama structures and manages reward engineering.

## 1. The Building Block: `RewardComponent`

At its core, RLlama encourages breaking down complex reward logic into smaller, manageable, and reusable pieces. This is achieved through the `RewardComponent` base class.

*   **Purpose:** To encapsulate the logic for calculating a *single aspect* of the total reward.
*   **Implementation:**
    *   You create a Python class that inherits from `rllama.rewards.RewardComponent`.
    *   You **must** implement the `calculate_reward` method.
    *   `calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float`:
        *   `raw_reward`: The original reward value returned directly by the environment step (often 0 or a simple task completion signal). You might use this, ignore it, or modify it.
        *   `info`: The `info` dictionary returned by the environment step (`env.step`). This is crucial for accessing environment-specific details not present in the standard observation/reward/terminated/truncated tuple (e.g., `is_success`, collision flags, distance to goal).
        *   `context`: A dictionary provided by the *user* via the `RewardShaper`. This allows passing arbitrary, dynamic information from your training loop into the reward calculation (e.g., `global_step`, `steps_in_episode`, agent's internal state, custom flags). See more on `context` below.
        *   `**kwargs`: Allows for future flexibility and passing additional standard arguments if needed.
        *   **Returns:** A single floating-point number representing the reward value calculated by *this specific component* for the current step.
*   **Example (`StepPenalty`):**
    ```python
    from rllama.rewards import RewardComponent

    class StepPenalty(RewardComponent):
        def __init__(self, penalty: float = -0.01):
            self.penalty = penalty

        def calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float:
            # This component ignores raw_reward, info, and context,
            # simply returns a fixed penalty for taking a step.
            return self.penalty
    ```
*   **Benefits:** Modularity, reusability, testability. Encourages clear separation of different reward sources (e.g., goal achievement vs. safety penalty vs. efficiency).

## 2. Combining Components: `RewardComposer`

Once you have individual components, you need a way to combine their outputs into a single, unweighted reward signal for the current step.

*   **Purpose:** To orchestrate the calculation and aggregation of rewards from multiple `RewardComponent` instances.
*   **Implementation:**
    *   Instantiate `RewardComposer` with a dictionary mapping unique string names to initialized `RewardComponent` objects.
    *   `composer = RewardComposer({"goal": GoalReward(), "penalty": StepPenalty(-0.05)})`
    *   Call the `compose` method during your RL step:
    *   `composed_reward_dict = composer.compose(raw_reward, info, context, **kwargs)`
        *   This method iterates through each registered component.
        *   It calls the `calculate_reward` method of each component, passing along the `raw_reward`, `info`, `context`, and `kwargs`.
        *   **Returns:** A dictionary where keys are the component names and values are the rewards calculated by each component for that step (e.g., `{"goal": 0.0, "penalty": -0.05}`). *Note: While the primary use case often involves summing these later, returning the dictionary allows for potential inspection or different aggregation strategies.*
*   **Benefits:** Separates the *what* (individual component logic) from the *how* (combining them). Makes it easy to add/remove/modify components without touching the core training loop logic significantly. Provides a clear overview of all contributing reward factors.

## 3. Dynamic Control: `RewardConfig` and `RewardShaper`

Static rewards are often insufficient. RLlama allows dynamic control over the *influence* of each component through configuration and the `RewardShaper`.

*   **`RewardConfig` (Conceptual / Data Structure):**
    *   **Purpose:** To declaratively define *how* reward components should be weighted and how those weights should change over time (scheduling).
    *   **Format:** Typically a Python dictionary (often loaded from YAML). It maps component names (matching those used in the `RewardComposer`) to their configuration.
    *   **Structure per Component:**
        ```yaml
        reward_shaping:
          component_name: # e.g., "step_penalty"
            # Optional: Parameters to initialize the component class
            params: { penalty: -0.01 }
            # Defines the weight and its schedule
            weight_schedule:
              initial_weight: 1.0 # Starting weight
              schedule_type: exponential # 'constant', 'exponential', 'linear' (future)
              # Parameters specific to schedule_type:
              decay_rate: 0.999 # For exponential decay
              decay_steps: 1 # How often (in global steps) to apply decay
              min_weight: 0.1 # Floor for the weight (optional)
              # For linear decay (example, might change):
              # end_weight: 0.1
              # decay_duration_steps: 10000
          # ... other components
        ```
*   **`RewardShaper`:**
    *   **Purpose:** The main interface in your training loop. It uses the `RewardComposer` and the `RewardConfig` to calculate the final, weighted, and potentially scheduled reward signal that the agent uses for learning. It also manages the state of the weight schedules.
    *   **Initialization:** `shaper = RewardShaper(composer, reward_shaping_config)`
    *   **Key Methods:**
        *   `shape(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float`:
            1.  Calls `composer.compose(...)` to get the dictionary of raw component rewards.
            2.  Retrieves the *current* weight for each component based on its schedule and the elapsed `global_step`.
            3.  Calculates the weighted sum: `final_reward = sum(component_reward * current_weight for component_reward, current_weight in ...)`
            4.  Returns the single `final_reward` float.
        *   `update_weights(self, global_step: int)`:
            *   This method **must** be called periodically in your training loop (usually once per `global_step`).
            *   It iterates through all components with schedules defined in the config.
            *   It updates the internal `current_weight` for each component based on its `schedule_type` and the provided `global_step`. For example, for exponential decay, it applies the `decay_rate` if `global_step` is a multiple of `decay_steps`.
        *   `get_current_weight(self, component_name: str) -> float`: Utility to inspect the current weight of a specific component (useful for logging).
*   **Benefits:** Enables dynamic reward strategies (curriculum learning, guidance fading), separates configuration from code, allows easy experimentation by modifying the config file.

## 4. The Power of `context`

The `context` dictionary, passed through `RewardShaper.shape` -> `RewardComposer.compose` -> `RewardComponent.calculate_reward`, is a key feature for advanced reward design.

*   **Purpose:** To inject arbitrary, step-dependent information from your training loop directly into the reward calculation logic of any component.
*   **Why it's powerful:** Standard `info` dictionaries are environment-specific. `context` allows you to pass information the environment doesn't know about, such as:
    *   Current `global_step` or `episode_num`.
    *   Steps taken *within* the current episode (`steps_in_episode`).
    *   Agent's internal state (e.g., uncertainty estimates, exploration progress).
    *   Flags indicating specific phases of training.
    *   Performance metrics calculated during the loop.
*   **Example Use Case:** A `RewardComponent` that gives a bonus only during the first 100 steps of an episode, using `context['steps_in_episode']`. Or a penalty that increases only if the agent has been stuck in the same region for too long (requires tracking state in the training loop and passing it via `context`).
*   **Implementation:** Simply populate a dictionary in your training loop before calling `shaper.shape` and pass it as the `context` argument. Components can then access these values within their `calculate_reward` method.

```python
# In training loop:
context = {
    "global_step": global_step,
    "steps_in_episode": steps_this_episode,
    "agent_uncertainty": agent.get_uncertainty(), # Fictional method
}
shaped_reward = shaper.shape(raw_reward, info, context)

# In a RewardComponent:
class UncertaintyPenalty(RewardComponent):
    def calculate_reward(self, raw_reward, info, context, **kwargs):
        uncertainty = context.get("agent_uncertainty", 0.0)
        # Penalize high uncertainty more
        return -uncertainty * 0.1
```



        