


          
Alright, let's create the content for the `docs/cookbook.md` file. This will contain practical examples and recipes for common reward engineering tasks using RLlama.

Here's the proposed content for the new file:

```markdown:/Users/cheencheen/Desktop/git/rl/RLlama/docs/cookbook.md
# RLlama Cookbook: Practical Recipes

This cookbook provides practical examples and patterns for implementing common reward engineering techniques using RLlama.

## Recipe 1: Basic Goal Reward + Step Penalty

This is the most common starting point. Reward the agent for reaching a goal and penalize it slightly for each step taken.

**Components:**

*   `GoalReward`: Provides a positive reward when a goal condition is met (e.g., `info['is_success'] == True`).
*   `StepPenalty`: Provides a small negative reward for every step.

**Configuration (`config.yaml`):**

```yaml
reward_shaping:
  goal:
    class: GoalReward
    params: { reward_value: 1.0 } # Or any positive value
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
  time_cost:
    class: StepPenalty
    params: { penalty: -0.01 } # Small negative value
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
```

**Implementation Notes:**

*   Ensure your environment's `info` dictionary contains a boolean flag (like `is_success`) that `GoalReward` can check. You might need to subclass `GoalReward` if your condition is different.
*   Adjust `reward_value` and `penalty` based on the expected episode length and desired behavior.

## Recipe 2: Sparse Reward with Potential-Based Shaping (Distance Reward)

In environments with sparse rewards (e.g., only getting a reward at the very end), potential-based reward shaping can provide denser guidance. Rewarding based on distance reduction is a common form.

**Components:**

*   `GoalReward`: For the final success signal.
*   `DistanceReward` (Custom): Calculates reward based on the change in distance to the goal.

**Custom Component (`distance_reward.py`):**

```python
import numpy as np
from rllama.rewards import RewardComponent

class DistanceReward(RewardComponent):
    def __init__(self, potential_scale: float = 1.0, gamma: float = 0.99):
        self.potential_scale = potential_scale
        self.gamma = gamma # Discount factor of the RL algorithm
        self.previous_potential = None

    def _calculate_potential(self, info: dict) -> float:
        """Calculates potential based on distance. Lower distance = higher potential."""
        agent_pos = info.get("agent_pos")
        target_pos = info.get("target_pos")
        if agent_pos is None or target_pos is None:
            return 0.0 # Or some default potential

        distance = np.linalg.norm(np.array(agent_pos) - np.array(target_pos))
        # Example potential: negative distance (closer is better/less negative)
        # Scale it to prevent overpowering other rewards
        potential = -distance * self.potential_scale
        return potential

    def calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float:
        current_potential = self._calculate_potential(info)

        shaping_reward = 0.0
        if self.previous_potential is not None:
            # Potential-based shaping formula: gamma * P(s') - P(s)
            shaping_reward = self.gamma * current_potential - self.previous_potential

        # Update previous potential for next step
        # Reset if episode ended (check context)
        terminated = context.get("terminated", False)
        truncated = context.get("truncated", False)
        if terminated or truncated:
            self.previous_potential = None # Reset for next episode start
        else:
            self.previous_potential = current_potential

        # This component ONLY provides the shaping term
        return shaping_reward

    # Optional: Add a reset method if needed outside of context checks
    # def reset(self):
    #    self.previous_potential = None
```

**Configuration (`config.yaml`):**

```yaml
reward_shaping:
  goal:
    class: GoalReward
    params: { reward_value: 1.0 }
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
  distance_shaping:
    class: DistanceReward # Your custom class
    params: { potential_scale: 0.1, gamma: 0.99 } # Adjust scale and gamma
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
```

**Implementation Notes:**

*   Requires `agent_pos` and `target_pos` (or similar) in the `info` dictionary.
*   The `gamma` parameter in the component should match the discount factor used by your RL agent for the theoretical guarantees of potential-based shaping to hold.
*   The `potential_scale` needs tuning; too high, and the agent might exploit the shaping term; too low, and it won't provide enough guidance.
*   Remember to pass `terminated` and `truncated` flags in the `context` dictionary when calling `shaper.shape`.

## Recipe 3: Curriculum Learning via Weight Scheduling

Gradually introduce or fade out reward components to guide the agent through different learning stages.

**Scenario:** Initially, strongly penalize collisions. As the agent learns to avoid them, reduce the penalty's weight and increase the weight of an efficiency reward (e.g., reaching the goal faster).

**Components:**

*   `GoalReward`
*   `CollisionPenalty` (Custom or built-in)
*   `EfficiencyBonus` (Custom, e.g., `1 / steps_in_episode` if goal reached)

**Configuration (`config.yaml`):**

```yaml
reward_shaping:
  goal:
    class: GoalReward
    params: { reward_value: 1.0 }
    weight_schedule: { initial_weight: 1.0, schedule_type: constant }
  safety:
    class: CollisionPenalty
    params: { penalty: -5.0 } # High initial penalty
    weight_schedule:
      initial_weight: 1.0
      schedule_type: exponential # Or linear
      decay_rate: 0.9999 # Slowly decrease weight
      decay_steps: 100 # Apply decay every 100 global steps
      min_weight: 0.1 # Maintain a small penalty
  speed:
    class: EfficiencyBonus
    params: { max_bonus: 0.5 }
    weight_schedule:
      initial_weight: 0.0 # Start with no efficiency bonus
      schedule_type: linear # Gradually increase weight
      end_weight: 1.0 # Target weight
      # Need start_step and duration for linear schedule (assuming shaper supports this)
      schedule_start_step: 10000 # Start increasing after 10k steps
      schedule_duration_steps: 50000 # Reach full weight over 50k steps
      # OR use exponential increase if preferred/supported
      # schedule_type: exponential
      # decay_rate: 1.0001 # Increase rate (use > 1)
      # decay_steps: 100
      # max_weight: 1.0
```

**Implementation Notes:**

*   Requires `CollisionPenalty` and `EfficiencyBonus` components to be defined.
*   The `RewardShaper` needs to support the specified `schedule_type` (e.g., `linear`, `exponential` increase/decrease). You might need to extend the shaper's scheduling logic if only constant and exponential decay are built-in.
*   Tune the schedule parameters (`decay_rate`, `decay_steps`, `min_weight`, `end_weight`, `schedule_start_step`, `schedule_duration_steps`) carefully based on the expected total training steps.
*   Remember to call `shaper.update_weights(global_step)` consistently in your training loop.

## Recipe 4: Using Context for State-Dependent Rewards

Modify rewards based on information not directly available in the environment's `info` dict, passed via the `context`.

**Scenario:** Give a bonus for exploration (visiting new states) only during the first half of training.

**Components:**

*   `ExplorationBonus` (Custom): Rewards visiting less frequent states.
*   Other components (GoalReward, StepPenalty, etc.)

**Custom Component (`exploration_bonus.py`):**

```python
from rllama.rewards import RewardComponent
from collections import defaultdict

class ExplorationBonus(RewardComponent):
    def __init__(self, bonus_scale: float = 0.01, max_training_steps: int = 1000000):
        self.bonus_scale = bonus_scale
        self.max_training_steps = max_training_steps
        self.state_visit_counts = defaultdict(int)

    def calculate_reward(self, raw_reward: float, info: dict, context: dict, **kwargs) -> float:
        global_step = context.get("global_step", 0)

        # Only apply bonus during the first half of training
        if global_step > self.max_training_steps / 2:
            return 0.0

        # Requires current state representation in context or info
        current_state = context.get("agent_state_representation")
        if current_state is None:
             # Try getting from info if available, otherwise cannot compute
             current_state = info.get("agent_state_representation")
             if current_state is None: return 0.0

        # Simple count-based exploration bonus: 1 / sqrt(count)
        # Needs a hashable state representation
        try:
            count = self.state_visit_counts[current_state]
            bonus = self.bonus_scale / np.sqrt(count + 1) # Add 1 to avoid division by zero
            self.state_visit_counts[current_state] += 1
            return bonus
        except TypeError: # State is not hashable (e.g., numpy array)
             print("Warning: State representation is not hashable for ExplorationBonus.")
             # Implement alternative state tracking if needed (e.g., discretization, hashing)
             return 0.0

    # Optional: Reset counts if needed between runs or phases
    # def reset_counts(self):
    #    self.state_visit_counts.clear()
```

**Configuration (`config.yaml`):**

```yaml
reward_shaping:
  # ... other components like goal, penalty ...
  explore:
    class: ExplorationBonus
    params: { bonus_scale: 0.01, max_training_steps: 1000000 } # Match total steps
    weight_schedule: { initial_weight: 1.0, schedule_type: constant } # Control via logic inside component
```

**Implementation Notes:**

*   The training loop **must** provide `global_step` and a hashable `agent_state_representation` in the `context` dictionary passed to `shaper.shape`.
*   The state visit count can grow large; consider alternatives for very large state spaces (e.g., pseudo-counts, hashing).
*   This component's logic controls *when* it's active based on `global_step`, even though its weight schedule is constant.

These recipes demonstrate the flexibility of RLlama. Combine and adapt these patterns to create sophisticated reward structures tailored to your specific RL problem. Remember to leverage the `context` dictionary and weight scheduling for dynamic and stateful reward logic.
```



        