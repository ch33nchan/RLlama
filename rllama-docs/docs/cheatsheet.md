

---

# RLlama Cheatsheet

## 🧠 Basic Usage

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward

# Create reward engine
engine = RewardEngine()

# Add component
engine.add_component(LengthReward(target_length=100))

# Set weights
engine.set_weights({"LengthReward": 1.0})

# Compute reward
context = {"response": "This is a test"}
reward = engine.compute(context)
```

---

## 🛠 Creating Custom Components

```python
from rllama.rewards.base import BaseReward

class MyReward(BaseReward):
    def __init__(self, param1=1.0):
        super().__init__()
        self.param1 = param1
    
    def compute(self, context):
        # Extract data from context
        data = context.get("my_key", None)
        
        # Calculate reward
        reward = self.calculate_my_reward(data)
        
        return reward
```

---

## 🔁 Common Patterns

### Stateful Components

```python
class StatefulReward(BaseReward):
    def __init__(self):
        super().__init__()
        self.previous_state = None
    
    def compute(self, context):
        current_state = context["state"]
        
        if self.previous_state is None:
            self.previous_state = current_state
            return 0.0
        
        # Calculate based on state change
        reward = calculate_improvement(self.previous_state, current_state)
        self.previous_state = current_state
        
        return reward
    
    def reset(self):
        self.previous_state = None
```

---

## 🎮 Gym Environment Integration

```python
import gym
from rllama.integration import GymWrapper

# Create environment
env = gym.make('CartPole-v1')

# Create reward engine with components
engine = RewardEngine()
engine.add_component(BalanceReward())
engine.add_component(CenteringReward())

# Wrap environment
wrapped_env = GymWrapper(engine).wrap(env)

# Use as normal gym environment
obs = wrapped_env.reset()
action = policy(obs)
obs, reward, done, info = wrapped_env.step(action)
```

---

## 🔧 Reward Optimization

```python
from rllama.rewards.optimizer import RewardOptimizer

def evaluate_weights(weights):
    engine.set_weights(weights)
    return run_evaluation(engine)

optimizer = RewardOptimizer(engine)
best_weights = optimizer.optimize(
    evaluate_weights,
    n_trials=50,
    search_space={
        "ComponentA": (0.1, 2.0),
        "ComponentB": (0.5, 5.0)
    }
)
```

---

## ⚙️ YAML Configuration

### config.yaml

```yaml
reward_components:
  - name: LengthReward
    params:
      target_length: 100
      strength: 0.5
  - name: DiversityReward
    params:
      history_size: 5
      strength: 1.0

weights:
  LengthReward: 0.3
  DiversityReward: 0.7
```

### Python

```python
from rllama.utils import load_config

config = load_config("config.yaml")
engine = RewardEngine.from_config(config)
```

---

## 🧾 Key Methods Reference

| Method                                   | Description                            |
| ---------------------------------------- | -------------------------------------- |
| `RewardEngine.add_component(component)`  | Add a reward component                 |
| `RewardEngine.set_weights(weights_dict)` | Set component weights                  |
| `RewardEngine.compute(context)`          | Calculate total reward                 |
| `RewardEngine.get_last_contributions()`  | Get component contributions            |
| `BaseReward.compute(context)`            | Main method to implement in components |
| `BaseReward.reset()`                     | Reset component's internal state       |
| `GymWrapper(engine).wrap(env)`           | Wrap a gym environment                 |

---

## 🐞 Debugging Tips

* Use `engine.get_last_contributions()` to see which components contribute what
* Add `print()` statements in your component’s `compute()` method
* Set component weights to `0` to isolate issues
* Use `RewardVisualizer(engine).plot_reward_history()` to visualize trends


