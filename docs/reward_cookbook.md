# RLlama Reward Shaping Cookbook

This guide explains how to use the `rllama.rewards` framework to design, combine, and dynamically shape reward signals for your Reinforcement Learning agents.

## Core Concepts

*   **Reward Components (`BaseReward`)**: The building blocks. Each represents a single source of reward (e.g., reaching a goal, action penalty). They are classes inheriting from `rllama.rewards.base.BaseReward`.
*   **Reward Composer (`RewardComposer`)**: Takes a list of reward components and calculates their individual raw values at each step. It also combines these raw values using weights to produce the final scalar reward.
*   **Reward Shaper (`RewardShaper` & `RewardConfig`)**: Manages the weights applied to each reward component. `RewardConfig` defines how a weight should behave (initial value, decay schedule, etc.), and `RewardShaper` applies these configurations over time.

## Getting Started: Basic Usage

```python
# Import necessary classes
from rllama.rewards.base import BaseReward
from rllama.rewards.common import StepPenaltyReward # Example common reward
from rllama.rewards.composition import RewardComposer
from rllama.rewards.shaping import RewardShaper, RewardConfig

# 1. Define your reward components (using common or custom)
class MyGoalReward(BaseReward):
    @property
    def name(self) -> str: return "my_goal"
    def __call__(self, state, action, next_state, info) -> float:
        return 1.0 if info.get("is_success", False) else 0.0

reward_components = [
    MyGoalReward(),
    StepPenaltyReward(penalty=-0.02)
]
composer = RewardComposer(reward_components)

# 2. Configure shaping (weights and schedules)
reward_configs = {
    "my_goal": RewardConfig(name="my_goal", initial_weight=1.0), # Constant weight
    "step_penalty": RewardConfig(name="step_penalty", initial_weight=0.5, decay_schedule='linear', decay_steps=10000, min_weight=0.05) # Linear decay
}
shaper = RewardShaper(reward_configs)

# 3. In your RL loop:
# state, action, next_state, info = ... (from env.step)
# global_step = ... (your step counter)

shaper.update_weights(global_step=global_step)
current_weights = shaper.get_weights()
raw_rewards = composer.compute_rewards(state, action, next_state, info)
final_reward = composer.combine_rewards(raw_rewards, current_weights)

# Use final_reward in your agent's update
# agent.learn(state, action, final_reward, next_state, done)