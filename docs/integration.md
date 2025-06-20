# Integration with RL Frameworks

RLlama is designed to integrate seamlessly with popular reinforcement learning frameworks. This guide covers:

- OpenAI Gym environments
- Stable Baselines3
- Custom RL environments

## OpenAI Gym Integration

```python
from rllama import RewardEngine
from rllama.integration import GymWrapper
import gym

# Create a standard Gym environment
env = gym.make('CartPole-v1')

# Create and configure a reward engine
engine = RewardEngine()
engine.add_component(ProgressReward(goal_pos=0.0))

# Create the wrapped environment
wrapped_env = GymWrapper(engine).wrap(env)

# Now use the wrapped environment with any RL algorithm
observation = wrapped_env.reset()
```

## Stable Baselines3 Integration

```python
from rllama.integration import StableBaselinesWrapper
from stable_baselines3 import PPO

# Create a wrapped environment
wrapped_env = GymWrapper(engine).wrap(env)

# Initialize a Stable Baselines model
model = PPO("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=50000)
```
