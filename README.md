
# RLlama: Your Framework for Advanced Reward Engineering in Reinforcement Learning

**Tired of messy reward code? Struggling to balance multiple objectives in RL? Need to automatically tune your reward signals? RLlama is here to help!**

RLlama is a powerful Python library designed to make **reward engineering** – the crucial process of designing the signals that guide your Reinforcement Learning agents – more **structured, flexible, scalable, and optimizable**. It's particularly well-suited for complex tasks like fine-tuning Large Language Models (LLMs) using Reinforcement Learning from Human Feedback (RLHF), but its principles apply to a wide range of RL problems.

---

## The Core Problem: Why is Reward Engineering So Hard?

Reinforcement Learning agents learn by maximizing a cumulative reward signal. The quality of this signal is paramount – a poorly designed reward can lead to agents learning the wrong behaviors, getting stuck, or failing to learn altogether.

Traditionally, crafting these rewards involves:

*   **Ad-hoc Code:** Mixing reward calculations directly into the main RL training loop.
*   **Manual Weight Tuning:** Guessing and checking weights for different reward components (e.g., "how much should I penalize staying still vs. reward reaching the goal?").
*   **Static Rewards:** Using fixed reward values that don't adapt as the agent learns.
*   **Difficulty Combining Sources:** Struggling to cleanly integrate diverse signals like task success, safety constraints, preference scores, and efficiency penalties.
*   **Reproducibility Issues:** Making it hard to track, share, and reuse effective reward strategies.

This becomes a major bottleneck, slowing down research and development. **RLlama tackles this challenge head-on.**

---

## The RLlama Solution: Core Concepts Explained

RLlama provides a set of building blocks and workflows to bring structure and power to your reward design process. Here are the fundamental ideas:

### 1. Composable Reward Components (`RewardComponent`)

*   **What it is:** The basic unit of reward calculation. Think of it as a small, focused function (encapsulated in a class) that calculates a specific piece of the total reward based on the environment's state or events. Examples:
    *   `GoalReward`: Gives a bonus when the agent reaches a goal state.
    *   `StepPenalty`: Applies a small cost for every action taken (encouraging efficiency).
    *   `CollisionPenalty`: Applies a penalty if the agent collides with an obstacle.
    *   `PreferenceScoreReward` (for RLHF): Uses the output of a preference model as a reward signal.
*   **How it works:** You create a Python class that inherits from `rllama.rewards.RewardComponent` and implement a `calculate_reward` method. This method receives information about the environment step (like the raw environment reward, `info` dictionary containing details, current state, action taken, etc.) and returns a single reward value for that specific component.
*   **Why it's useful:**
    *   **Modularity:** Breaks down complex reward logic into manageable, understandable pieces.
    *   **Reusability:** Easily reuse components across different projects or experiments.
    *   **Testability:** Test each reward component in isolation.

### 2. Declarative Composition (`RewardComposer`)

*   **What it is:** The orchestrator that combines the outputs of multiple `RewardComponent` instances.
*   **How it works:** You provide the `RewardComposer` with a collection (usually a dictionary) of named `RewardComponent` objects. When its `compose` method is called during an RL step, it iterates through each component, calls its `calculate_reward` method, and aggregates the results. By default, it simply sums the rewards from all components, but it's designed to potentially support other aggregation strategies (like averaging or weighted sums) in the future.
*   **Why it's useful:**
    *   **Separation of Concerns:** Keeps the logic of *combining* rewards separate from the logic of *calculating* individual rewards.
    *   **Flexibility:** Easily add, remove, or swap components without changing the core RL loop.
    *   **Clarity:** Makes the overall reward structure explicit.

### 3. Configurable Shaping & Scheduling (`RewardConfig`, `RewardShaper`)

*   **What it is:** This is where the dynamic adaptation happens. It allows you to control the *influence* (weight) of each reward component and change that influence over time (scheduling).
    *   `RewardConfig`: Represents the configuration for shaping. Typically defined in a YAML file or Python dictionary. It specifies which components to use, their initial weights, and parameters for scheduling their weights.
    *   `RewardShaper`: Takes the `RewardComposer` and the `RewardConfig`. Its main job is to:
        1.  Calculate the *weighted* sum of rewards from the composer based on the current weights defined in the config.
        2.  Update the weights of components over time according to the specified schedules (e.g., make a penalty harsher or a guidance reward fade out as the agent learns).
*   **How it works:**
    *   **Configuration:** You define a structure (like the example below) specifying parameters for each component:
        *   `initial_weight`: The starting importance of the component.
        *   `schedule_type`: How the weight changes (e.g., `exponential`, `linear`, `constant`). Currently, `exponential` decay and `constant` are the primary focus.
        *   `decay_rate` (for exponential): Multiplier applied each step/episode (e.g., 0.999 slowly reduces weight).
        *   `decay_steps`: How often to apply the decay (e.g., apply decay every 1000 steps).
        *   `min_weight`: A floor for the weight, preventing it from decaying to zero if needed.
    *   **Shaping (`shape` method):** In your RL loop, you call `shaper.shape(raw_reward, info, context)`. It gets the composed reward, applies the *current* weights to each component's contribution (implicitly, as the composer just sums them, the shaper applies the *overall* weight logic based on config, though future versions might allow per-component weighting application *within* the shaper), and returns the final shaped reward for the agent to learn from. The `context` dictionary is crucial – it allows you to pass *extra* information (like current training step, episode number, agent's internal state) to the shaping logic, enabling very sophisticated, state-dependent reward adjustments beyond simple time-based schedules.
    *   **Weight Updates (`update_weights` method):** You periodically call `shaper.update_weights(global_step)` (or similar) to advance the schedules based on the number of steps or episodes elapsed.
*   **Why it's useful:**
    *   **Dynamic Adaptation:** Rewards are no longer static! Guide the agent differently at various stages of training (e.g., strong guidance early on, fading later).
    *   **Curriculum Learning:** Implement simple curricula by scheduling reward components.
    *   **Declarative Control:** Define complex scheduling behavior via simple configuration parameters, not complex code.
    *   **Experimentation:** Easily tweak weights and schedules by changing the configuration file.

### 4. Integrated Hyperparameter Optimization (`BayesianRewardOptimizer`)

*   **What it is:** Finding the *best* weights and schedule parameters manually is tedious and often suboptimal. This component automates the search using Bayesian Optimization.
*   **How it works:**
    *   Leverages the powerful `Optuna` library.
    *   You define a `search_space` specifying which parameters in your `RewardConfig` you want to tune (e.g., `goal_reward_initial_weight`, `step_penalty_decay_rate`) and the range or choices for each.
    *   You provide an `objective` function. This function takes a set of suggested parameters from Optuna, runs a full RL training/evaluation loop using those parameters to configure the `RewardShaper`, and returns a performance metric (e.g., average return over the last 10 episodes).
    *   The `BayesianRewardOptimizer` repeatedly calls your `objective` function with different parameter combinations, intelligently exploring the search space to find the set of parameters that maximizes (or minimizes) your performance metric.
*   **Why it's useful:**
    *   **Automation:** Saves significant manual effort in tuning reward parameters.
    *   **Better Performance:** Often finds better parameter combinations than manual tuning.
    *   **Data-Driven Decisions:** Bases reward design choices on actual agent performance.
    *   **Handles Complexity:** Efficiently searches spaces with multiple interacting parameters.

---

## Installation

Get started with RLlama using pip:

```bash
# Ensure you have Python 3.8+ and pip installed
pip install rllama
```
*(Note: This assumes the package is available on PyPI or you are installing from a local source distribution where packaging (`setup.py` or `pyproject.toml`) is correctly configured.)*

---

## Quick Start: Putting it Together

Let's see a conceptual example of how these pieces fit into a typical RL training loop.

```python
import gymnasium as gym
from rllama.rewards import RewardComposer, RewardShaper, GoalReward, StepPenalty # Your components
from rllama.config import load_reward_config # Helper to load YAML
# Assume other necessary imports for your RL agent and training loop

# 1. Load Reward Configuration from YAML (or define as dict)
# Example optimizer_demo_config.yaml content:
# reward_shaping:
#   goal_reward:
#     class: GoalReward # You'll need a way to map this string to the class
#     params: { target_reward: 1.0 }
#     weight_schedule: { initial_weight: 10.0, schedule_type: constant }
#   step_penalty:
#     class: StepPenalty
#     params: { penalty: -0.01 }
#     weight_schedule: { initial_weight: 1.0, schedule_type: exponential, decay_rate: 0.9995, decay_steps: 1 }
# training: # Other training params
#   num_episodes: 5000
#   max_steps_per_episode: 200

config_path = "path/to/your/reward_config.yaml"
config = load_reward_config(config_path)
reward_shaping_config = config.get("reward_shaping", {})
training_config = config.get("training", {})

# 2. Instantiate Components (Example using a simple factory/registry pattern)
component_registry = {"GoalReward": GoalReward, "StepPenalty": StepPenalty} # Map names to classes
components = {}
for name, comp_config in reward_shaping_config.items():
    cls_name = comp_config.get("class")
    params = comp_config.get("params", {})
    if cls_name in component_registry:
        components[name] = component_registry[cls_name](**params)
    else:
        print(f"Warning: Component class '{cls_name}' not found in registry.")

# 3. Create Composer and Shaper
composer = RewardComposer(components)
# The Shaper needs the configuration to manage weights and schedules
shaper = RewardShaper(composer, reward_shaping_config)

# 4. Setup Environment and Agent
env = gym.make("FrozenLake-v1") # Or your custom environment
# agent = YourAgent(...) # Instantiate your RL agent

# 5. The RL Training Loop with RLlama Integration
num_episodes = training_config.get("num_episodes", 1000)
max_steps_episode = training_config.get("max_steps_per_episode", 200)
global_step = 0

print("Starting training...")
for episode in range(num_episodes):
    state, info = env.reset()
    terminated = truncated = False
    episode_shaped_reward = 0
    steps_in_episode = 0

    while not terminated and not truncated:
        # --- RLlama: Update Weights Based on Schedule ---
        # Call this at the start of the step or episode, depending on schedule logic
        shaper.update_weights(global_step=global_step)

        # Agent selects action based on current state
        action = agent.select_action(state)

        # Environment steps forward
        next_state, raw_reward, terminated, truncated, info = env.step(action)

        # --- RLlama: Calculate Shaped Reward ---
        # Prepare context (optional but powerful!)
        shaper_context = {
            "steps_in_episode": steps_in_episode,
            "global_step": global_step,
            "max_steps_episode": max_steps_episode,
            "is_goal_reached": info.get("is_success", False), # Example context item
            # Add any other relevant info from the environment or agent
        }
        # Get the final reward signal to train the agent
        shaped_reward = shaper.shape(raw_reward, info, shaper_context)
        # --- End RLlama Integration ---

        # Agent learns using the shaped reward
        agent.update(state, action, shaped_reward, next_state, terminated)

        # Update state and counters
        state = next_state
        episode_shaped_reward += shaped_reward
        steps_in_episode += 1
        global_step += 1

        if steps_in_episode >= max_steps_episode:
            truncated = True # Ensure termination if max steps reached

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Shaped Reward: {episode_shaped_reward:.2f}, Current Step Penalty Weight: {shaper.get_current_weight('step_penalty'):.4f}") # Example logging

env.close()
print("Training finished.")

# To automatically find the best 'initial_weight', 'decay_rate' etc.,
# you would wrap this loop inside an 'objective' function and use
# the BayesianRewardOptimizer. See the Optimization Guide for details.
```

---

## Why Use RLlama? Key Features & Benefits

*   **Modular & Reusable Reward Logic:** Build rewards like Lego blocks. Define once, reuse everywhere. Makes code cleaner and easier to maintain.
*   **Declarative Configuration:** Define complex reward strategies in simple YAML or dictionary formats. Easy to read, modify, and share. Separates reward *design* from RL algorithm *implementation*.
*   **Powerful Dynamic Shaping:** Go beyond static rewards. Implement adaptive rewards, curriculum learning, and guidance fading effortlessly using built-in scheduling. Use the `context` dictionary for fine-grained, state-dependent adjustments.
*   **Automated Reward Tuning:** Stop guessing! Leverage Bayesian Optimization (`BayesianRewardOptimizer` + Optuna) to automatically find high-performing reward weights and schedule parameters based on actual results.
*   **Extensibility:** Designed to be extended. Easily create your own custom `RewardComponent` classes tailored to your specific environment or task needs.
*   **Improved Reproducibility:** Explicit configuration and modular design make experiments easier to reproduce and compare.

---

## Dive Deeper: Documentation

Ready to master reward engineering? Explore the detailed documentation:

*   **[Concepts](./docs/concepts.md):** A deep dive into `RewardComponent`, `RewardComposer`, `RewardConfig`, `RewardShaper`, scheduling, and the philosophy behind RLlama. *(Link assumes file exists)*
*   **[Usage Guide / API](./docs/usage.md):** Practical examples and API reference for using the core classes and provided components. *(Link assumes file exists)*
*   **[Optimization Guide](./docs/optimization_guide.md):** Step-by-step instructions on setting up and running the `BayesianRewardOptimizer` with Optuna. *(Link assumes file exists)*
*   **[Cookbook](./docs/cookbook.md):** Ready-to-use recipes for common reward shaping patterns and scenarios (e.g., combining penalties, implementing decay, full optimization workflow). *(Link assumes file exists)*

---

## Examples in Action

See RLlama applied in practice:

*   **[`/examples/optimizer_demo.py`](./examples/optimizer_demo.py):** The primary example showcasing the full workflow: loading configuration, defining an objective function for Optuna, using `BayesianRewardOptimizer` to tune weights and decay rates for FrozenLake, and setting up result visualization. A great starting point!
*   *(Add links to other key examples as they are developed, e.g., an example focusing purely on composition, or one demonstrating custom components)*

---

## Contributing

RLlama is an evolving project. Contributions, suggestions, and bug reports are highly welcome! Please refer to `CONTRIBUTING.md` (if available) for guidelines or open an issue on the project repository to discuss your ideas.
```


        