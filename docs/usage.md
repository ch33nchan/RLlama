# RLlama Usage Guide & API Reference

This guide provides practical steps and examples for integrating RLlama into your Reinforcement Learning projects, along with a reference for the core API components.

## 1. Installation

If you haven't already, install RLlama using pip:

```bash
pip install rllama # Or pip install . if installing from local source
```

Ensure you have Python 3.8+ and the necessary dependencies (like `PyYAML` for config loading, `numpy` might be implicitly used).

## 2. Defining Custom Reward Components

The foundation of RLlama is the `RewardComponent`. You'll often define your own components tailored to your specific environment or task.

**Steps:**

1.  Import the base class: `from rllama.rewards import RewardComponent`
2.  Create a new class inheriting from `RewardComponent`.
3.  Implement the `__init__` method (optional) to accept any necessary parameters (e.g., target values, penalty amounts).
4.  Implement the `calculate_reward` method. This is where your core reward logic resides.

**Example: `DistanceReward`**

Let's create a component that rewards the agent for getting closer to a target location, using information passed via the `info` dictionary.

```python
import numpy as np
from rllama.rewards import RewardComponent

class DistanceReward(RewardComponent):
    """
    Calculates a reward based on the change in distance to a target.
    Assumes 'agent_pos' and 'target_pos' are in the info dict.
    """
    def __init__(self, reward_scale: float = 1.0):
        self.reward_scale = reward_scale
        self.previous_distance = None

    def calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float:
        agent_pos = info.get("agent_pos")
        target_pos = info.get("target_pos")

        if agent_pos is None or target_pos is None:
            # Cannot calculate if positions are missing
            return 0.0

        current_distance = np.linalg.norm(np.array(agent_pos) - np.array(target_pos))

        reward = 0.0
        if self.previous_distance is not None:
            # Reward is positive if distance decreased
            reward = (self.previous_distance - current_distance) * self.reward_scale

        # Store current distance for the next step's calculation
        self.previous_distance = current_distance

        # Reset previous distance if the episode ended (using context)
        # Assumes 'terminated' or 'truncated' is passed in context
        if context.get("terminated", False) or context.get("truncated", False):
             self.previous_distance = None # Ready for next episode

        return reward

```

## 3. Configuration (`RewardConfig`)

Define how your components are used, weighted, and scheduled. This is typically done in a YAML file or a Python dictionary.

**YAML Example (`my_reward_config.yaml`):**

```yaml
reward_shaping:
  goal_reached:
    class: GoalReward # Assumes a GoalReward component exists
    params: { reward_value: 100.0 }
    weight_schedule:
      initial_weight: 1.0
      schedule_type: constant # Weight never changes

  step_cost:
    class: StepPenalty
    params: { penalty: -0.1 }
    weight_schedule:
      initial_weight: 1.0
      schedule_type: exponential
      decay_rate: 0.9999
      decay_steps: 1 # Decay every step
      min_weight: 0.1 # Don't let penalty vanish completely

  distance_guidance:
    class: DistanceReward # Our custom component
    params: { reward_scale: 0.5 }
    weight_schedule:
      initial_weight: 5.0 # Strong guidance initially
      schedule_type: exponential
      decay_rate: 0.999
      decay_steps: 10 # Decay less frequently
      min_weight: 0.0 # Allow guidance to fade out

# Optional: Other training parameters can be included
training:
  num_episodes: 10000
  max_steps_per_episode: 500
```

**Loading Configuration:**

Use helper functions (or standard YAML libraries) to load this into a Python dictionary.

```python
import yaml
from rllama.config import load_reward_config # Assuming this helper exists

# Option 1: Using a helper (if provided by rllama.config)
# config = load_reward_config("my_reward_config.yaml")

# Option 2: Using PyYAML directly
with open("my_reward_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

reward_shaping_config = config.get("reward_shaping", {})
```

## 4. Instantiating Components, Composer, and Shaper

You need to map the class names in your config to the actual Python classes and then instantiate the core RLlama objects.

```python
from rllama.rewards import RewardComposer, RewardShaper, StepPenalty, GoalReward # Built-ins
# Import your custom components
from your_module import DistanceReward

# --- Component Instantiation (Factory/Registry Pattern Recommended) ---
# Map string names from config to actual classes
component_registry = {
    "GoalReward": GoalReward,
    "StepPenalty": StepPenalty,
    "DistanceReward": DistanceReward,
    # Add other components here
}

components = {}
for name, comp_config in reward_shaping_config.items():
    cls_name = comp_config.get("class")
    params = comp_config.get("params", {})
    if cls_name in component_registry:
        try:
            components[name] = component_registry[cls_name](**params)
            print(f"Instantiated component: {name} ({cls_name}) with params: {params}")
        except Exception as e:
            print(f"Error instantiating {name} ({cls_name}): {e}")
    else:
        print(f"Warning: Component class '{cls_name}' for '{name}' not found in registry.")
# --- End Component Instantiation ---

# --- Create Composer and Shaper ---
if not components:
    raise ValueError("No reward components were successfully instantiated.")

composer = RewardComposer(components)
shaper = RewardShaper(composer, reward_shaping_config) # Pass the config dict here
print("RewardComposer and RewardShaper created.")
# --- End Composer/Shaper Creation ---
```

## 5. Integrating into the Training Loop

Modify your RL training loop to use the `RewardShaper`.

```python
import gymnasium as gym
# ... other imports (agent, config loading, components, composer, shaper from above)

# Assume env, agent, config, shaper are initialized as shown previously
training_config = config.get("training", {})
num_episodes = training_config.get("num_episodes", 1000)
max_steps_episode = training_config.get("max_steps_per_episode", 200)
global_step = 0

print("Starting training...")
for episode in range(num_episodes):
    state, info = env.reset()
    # Reset any stateful components if necessary (e.g., previous_distance in DistanceReward)
    # This might be handled internally if components check for context["terminated"]
    # or you might need an explicit reset method on the shaper/composer/components.

    terminated = truncated = False
    episode_shaped_reward = 0
    steps_in_episode = 0

    while not terminated and not truncated:
        # 1. Update Weights (CRITICAL!)
        # Call this BEFORE calculating the shaped reward for the step
        shaper.update_weights(global_step=global_step)

        # Agent selects action
        action = agent.select_action(state) # Your agent's policy

        # Environment step
        next_state, raw_reward, terminated, truncated, info = env.step(action)

        # 2. Prepare Context (Optional but Recommended)
        # Pass any extra info needed by components or future shaping logic
        context = {
            "global_step": global_step,
            "episode_num": episode,
            "steps_in_episode": steps_in_episode,
            "terminated": terminated,
            "truncated": truncated,
            # Add environment-specific or agent-specific info if needed
            # e.g., "agent_pos": info.get("agent_pos"), "target_pos": info.get("target_pos")
        }

        # 3. Calculate Shaped Reward (The Core Integration)
        shaped_reward = shaper.shape(raw_reward, info, context)

        # 4. Agent Update
        # Use the SHAPED reward for learning
        agent.update(state, action, shaped_reward, next_state, terminated or truncated)

        # Updates for next iteration
        state = next_state
        episode_shaped_reward += shaped_reward
        steps_in_episode += 1
        global_step += 1

        if steps_in_episode >= max_steps_episode:
            truncated = True # Enforce max steps

    # Logging example
    if episode % 100 == 0:
        log_msg = f"Episode {episode}: Shaped Reward={episode_shaped_reward:.2f}"
        # Log current weights for debugging/analysis
        for name in components.keys():
             log_msg += f", W_{name}={shaper.get_current_weight(name):.4f}"
        print(log_msg)

env.close()
print("Training finished.")
```

## 6. API Reference

### `rllama.rewards.RewardComponent` (Base Class)

*   **`__init__(self, ...)`**: Optional constructor to store parameters.
*   **`calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float`**: **(Abstract Method - Must be implemented by subclasses)**. Calculates and returns the reward value for this component based on environment step results and context.

### `rllama.rewards.RewardComposer`

*   **`__init__(self, components: Dict[str, RewardComponent])`**:
    *   `components`: A dictionary mapping unique string names to initialized `RewardComponent` instances.
*   **`compose(self, raw_reward: float, info: dict, context: dict, **kwargs) -> Dict[str, float]`**:
    *   Calls `calculate_reward` on each registered component.
    *   Returns a dictionary mapping component names to their calculated reward values for the current step.

### `rllama.rewards.RewardShaper`

*   **`__init__(self, composer: RewardComposer, config: dict)`**:
    *   `composer`: An initialized `RewardComposer` instance.
    *   `config`: The reward shaping configuration dictionary (loaded from YAML or defined directly), containing component parameters and weight schedules under a key like `reward_shaping`.
*   **`shape(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float`**:
    *   Orchestrates the reward calculation: calls `composer.compose`, applies current weights based on schedules, and returns the final aggregated (weighted sum) reward value for the agent.
*   **`update_weights(self, global_step: int)`**:
    *   Updates the internal current weights of all components based on their defined schedules and the provided `global_step`. **Must be called periodically (e.g., every step) in the training loop.**
*   **`get_current_weight(self, component_name: str) -> float`**:
    *   Returns the current calculated weight for the specified component name. Useful for logging and debugging.
*   **`get_component_rewards(self) -> Dict[str, float]`**: (Optional - might exist)
    *   Returns the dictionary of unweighted rewards from the last call to `composer.compose` (useful for detailed logging).

This guide provides the essential steps to use RLlama. For automating the tuning of configuration parameters (weights, decay rates), refer to the `Optimization Guide`.
```


        