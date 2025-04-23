
# RLlama

*A framework for exploring and implementing Reinforcement Learning concepts.*

This repository currently focuses on providing a modular and configurable framework for **Reward Shaping** in RL environments.

## Reward Shaping Framework

This framework allows you to easily define, combine, and dynamically adjust multiple reward components to guide your RL agent's learning process.

### Features

*   ✅ **Modular Components:** Define individual reward sources (goal, penalty, novelty, etc.) as reusable classes inheriting from `BaseReward`.
*   ✅ **Composition:** Combine multiple reward components using `RewardComposer`.
*   ✅ **Dynamic Weighting:** Adjust the influence of each component over time using `RewardShaper` and flexible `RewardConfig` settings (constant, linear/exponential decay).
*   ✅ **YAML Configuration:** Define components, parameters, and shaping schedules externally via YAML files for easy experimentation.
*   ✅ **Component Registry:** Instantiate reward components by name, facilitating configuration-driven setup.
*   ✅ **Normalization:** Optionally normalize raw reward values within the `RewardComposer` to stabilize learning.
*   ✅ **Visualization:** Log and visualize reward component weights and raw values over time using the `RewardDashboard` (outputs interactive Plotly HTML).
*   ⏳ **Optimization (Planned):** Includes a placeholder for Bayesian Optimization (`BayesianRewardOptimizer`) to automatically tune reward parameters (implementation pending).
*   ✅ **Demo:** Includes an example (`examples/reward_integration_demo.py`) showcasing integration with Gymnasium and a simple Q-Learning agent.

### Core Concepts

*   **Reward Components (`BaseReward`)**: The building blocks. Each represents a single source of reward (e.g., reaching a goal, action penalty). They are classes inheriting from `rllama.rewards.base.BaseReward`.
*   **Reward Composer (`RewardComposer`)**: Takes a list of reward components and calculates their individual raw values at each step. It can optionally normalize these values and combines them using weights to produce the final scalar reward.
*   **Reward Shaper (`RewardShaper` & `RewardConfig`)**: Manages the weights applied to each reward component. `RewardConfig` defines how a weight should behave (initial value, decay schedule, etc.), and `RewardShaper` applies these configurations over time based on the global training step.

### Getting Started: Basic Usage

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
    StepPenaltyReward(penalty=-0.02) # Name is 'step_penalty' by default
]
# Composer calculates raw rewards from components
composer = RewardComposer(reward_components)

# 2. Configure shaping (weights and schedules)
# Keys MUST match the .name property of the components
reward_configs = {
    "my_goal": RewardConfig(name="my_goal", initial_weight=1.0), # Constant weight
    "step_penalty": RewardConfig(name="step_penalty", initial_weight=0.5, decay_schedule='linear', decay_steps=10000, min_weight=0.05) # Linear decay
}
# Shaper manages the weights based on configs
shaper = RewardShaper(reward_configs)

# 3. In your RL loop:
# state, action, next_state, info = ... (from env.step)
# global_step = ... (your step counter)

# Update weights based on the current step
shaper.update_weights(global_step=global_step)
current_weights = shaper.get_weights()

# Calculate raw rewards for this transition
raw_rewards = composer.compute_rewards(state, action, next_state, info)

# Combine raw rewards using current weights
final_reward = composer.combine_rewards(raw_rewards, current_weights)

# Use final_reward in your agent's update
# agent.learn(state, action, final_reward, next_state, done)
```

### Configuration via YAML

Instead of defining `RewardConfig` objects directly in Python, you can load the shaping configuration from a YAML file. This allows easier experimentation without modifying the training script. The demo script (`examples/reward_integration_demo.py`) uses this approach.

**Example (`examples/reward_config.yaml`):**

```yaml
# Global composer settings (optional)
composer_settings:
  normalize: false # Set to true to enable running normalization
  norm_window: 1000

# Define which reward components to use and their parameters
# Used by the registry to instantiate components
reward_components:
  goal: # Name used in registry or as key if 'class' is specified
    class: frozen_lake_goal # Specific class name (registered or imported)
    params: {} # Parameters for FrozenLakeGoalReward.__init__
  hole:
    class: frozen_lake_hole
    params: {}
  step:
    class: step_penalty # Use registered name
    params:
      penalty: -0.01 # Parameter for StepPenaltyReward.__init__

# Define shaping parameters
# Keys here MUST match the 'name' property of the instantiated reward components
reward_shaping:
  frozen_lake_goal: # Must match FrozenLakeGoalReward().name
    initial_weight: 1.0
    decay_schedule: 'none'
  frozen_lake_hole: # Must match FrozenLakeHolePenalty().name
    initial_weight: 1.0
    decay_schedule: 'none'
  step_penalty: # Must match StepPenaltyReward().name
    initial_weight: 1.0
    # Example with decay:
    # decay_schedule: 'linear'
    # decay_steps: 5000
    # min_weight: 0.0
```

**Loading in Python (Simplified Example):**

```python
import yaml
from rllama.rewards.shaping import RewardShaper, RewardConfig
from rllama.rewards.composition import RewardComposer
from rllama.rewards.registry import create_reward_component
# ... other imports ...

config_path = 'examples/reward_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Instantiate components using the registry
reward_components_config = config.get('reward_components', {})
reward_components = []
for name, comp_config in reward_components_config.items():
    class_name = comp_config.get('class', name)
    params = comp_config.get('params', {})
    component = create_reward_component(class_name, **params)
    reward_components.append(component)

# Initialize Composer
composer_settings = config.get('composer_settings', {})
composer = RewardComposer(reward_components, **composer_settings)

# Configure Shaper
shaping_configs_dict = config.get('reward_shaping', {})
reward_configs = {name: RewardConfig(**cfg_dict) for name, cfg_dict in shaping_configs_dict.items()}
shaper = RewardShaper(reward_configs)

# ... rest of your RL setup ...
```

### Reward Component Registry

The `rllama.rewards.registry` module allows you to instantiate reward components by name, as shown in the YAML loading example above. This decouples your main script from specific component classes.

```python
from rllama.rewards.registry import create_reward_component, register_reward_component
from rllama.rewards.base import BaseReward

# Common rewards are often pre-registered
# from rllama.rewards.common import StepPenaltyReward

# Register custom components if needed
class MyCustomReward(BaseReward):
    @property
    def name(self) -> str: return "my_custom"
    def __call__(self, *args, **kwargs) -> float: return 0.0
register_reward_component("my_custom_reward", MyCustomReward)

# Instantiate from name and parameters (e.g., loaded from YAML)
component_name = "step_penalty"
params = {"penalty": -0.05}
step_penalty_component = create_reward_component(component_name, **params)

# Use in RewardComposer
# composer = RewardComposer([step_penalty_component, ...])
```

### Visualization (`RewardDashboard`)

The `rllama.rewards.visualization.RewardDashboard` class helps visualize reward dynamics during training.

1.  **Instantiate:** `dashboard = RewardDashboard()`
2.  **Log Data:** In your training loop, after calculating rewards and weights:
    `dashboard.log_iteration(weights=current_weights, metrics=raw_rewards, step=global_step)`
3.  **Generate Report:** After training:
    `dashboard.generate_dashboard(output_file="reward_analysis.html")`

This creates an interactive HTML file (using Plotly) plotting weights and raw reward values over time. See the demo script for an example.

### Optimization (`BayesianRewardOptimizer`)

*(Placeholder)* The `rllama.rewards.optimization.BayesianRewardOptimizer` class is intended to help automatically find good `RewardConfig` parameters (weights, decay schedules) by running experiments. Its implementation using libraries like Optuna is planned for the future.

### Running the Demo

The example script `examples/reward_integration_demo.py` demonstrates the framework's features using the FrozenLake environment from Gymnasium and a simple Q-Learning agent.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or manually: pip install gymnasium numpy pyyaml plotly pandas matplotlib
    ```
2.  **Run:**
    ```bash
    python examples/reward_integration_demo.py
    ```
3.  **Observe:** The script will print training progress and open a Gymnasium render window.
4.  **Analyze:** After completion, check the generated files in the `examples/` directory:
    *   `frozen_lake_reward_dashboard.html`: Interactive plot of reward weights and metrics.
    *   `episode_rewards_plot.png`: Plot of total episode rewards over time.

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd RLlama

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Future Work (Ideas)


*   Implement `BayesianRewardOptimizer`.
*   Add more common reward components.
*   Integrate with standard RL libraries (e.g., Stable Baselines3).
*   Add comprehensive unit tests.



