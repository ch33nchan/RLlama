
Reinforcement Learning Without RLlama
To understand the value RLlama provides, let's examine what reward engineering typically looks like without such a framework.

Traditional Reward Function Implementation
In standard reinforcement learning implementations, reward functions are often implemented as monolithic functions that combine multiple objectives:

Python
def calculate_reward(state, action, next_state, done):
    # Task completion reward
    task_reward = 10.0 if done and is_task_completed(next_state) else 0.0
    
    # Penalty for collisions
    collision_penalty = -5.0 if detect_collision(next_state) else 0.0
    
    # Time penalty to encourage efficiency
    time_penalty = -0.01  # Small penalty for each time step
    
    # Energy usage penalty
    energy_used = calculate_energy(action)
    energy_penalty = -0.05 * energy_used
    
    # Exploration bonus
    novelty_score = calculate_novelty(next_state)
    exploration_bonus = 0.2 * novelty_score
    
    # Progress reward
    progress = calculate_progress(state, next_state)
    progress_reward = 0.3 * progress
    
    # Combine all rewards
    total_reward = (task_reward + 
                    collision_penalty + 
                    time_penalty + 
                    energy_penalty + 
                    exploration_bonus + 
                    progress_reward)
    
    return total_reward
Problems with This Approach
1. Hard-Coded Balance Between Components
In the example above, the weights (10.0, -5.0, -0.01, etc.) are hard-coded. Tuning these values requires:

Manually changing values
Retraining the agent
Evaluating performance
Repeating until satisfactory
This process can take days or weeks of trial and error.

2. Limited Insight into Reward Contributions
When your agent receives a total reward of -2.3, it's unclear which components contributed most significantly:

Was there a collision?
Did the agent make progress but use too much energy?
Is the time penalty dominating the reward?
Without additional logging code, these questions are hard to answer.

3. Difficult Maintenance and Iteration
Adding a new reward component requires:

Modifying the core reward function
Carefully balancing it against existing components
Potentially retuning all weights
Retraining the agent completely
4. Code Duplication Across Projects
Common reward patterns (like progress rewards or efficiency penalties) are reimplemented for each new project, leading to:

Repeated code
Inconsistent implementations
Time wasted solving the same problems
5. Challenging Debugging
When an agent behaves unexpectedly:

It's difficult to isolate which reward component is responsible
There's no easy way to experiment with removing or modifying specific components
Logging reward contributions requires custom code
A Concrete Example: Robot Navigation
Consider training a robot to navigate to a target while avoiding obstacles:

Python
def navigation_reward(state, action, next_state, done):
    # Distance to target
    prev_distance = euclidean_distance(state['position'], TARGET_POSITION)
    curr_distance = euclidean_distance(next_state['position'], TARGET_POSITION)
    distance_improvement = prev_distance - curr_distance
    
    # Goal reached bonus
    goal_reached = 0
    if curr_distance < GOAL_THRESHOLD:
        goal_reached = 100
    
    # Collision penalty
    collision = -50 if detect_collision(next_state) else 0
    
    # Energy efficiency
    energy_used = calculate_energy(action)
    energy_penalty = -0.1 * energy_used
    
    # Smoothness of motion
    action_smoothness = -0.2 * calculate_jerk(state['velocity'], next_state['velocity'])
    
    # Final reward
    reward = (
        1.5 * distance_improvement +  # Progress weight
        goal_reached +
        collision +
        energy_penalty +
        action_smoothness
    )
    
    return reward
Debugging Challenges
If the robot behaves strangely (e.g., moves in circles or freezes):

Is the distance reward encouraging local minima?
Is the energy penalty too high, causing inaction?
Is the smoothness term interfering with necessary quick movements?
Are the relative scales of different components appropriate?
Without a framework like RLlama, answering these questions requires:

Adding custom logging
Manually tracking reward components
Experimenting with different weight combinations through guesswork
Potentially retraining many times
In the next section, we'll see how RLlama transforms this process. EOF

cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/with-rllama.md << 'EOF'

Reinforcement Learning With RLlama
Now let's see how the same reward engineering problems are addressed using RLlama's structured approach.

Modular Reward Component Design
RLlama enables you to define each reward component as a separate class:

Python
from rllama import RewardEngine
from rllama.rewards.base import BaseReward

# Define reusable reward components
class DistanceReward(BaseReward):
    def __init__(self, target_position, strength=1.0):
        super().__init__()
        self.target_position = target_position
        self.strength = strength
    
    def compute(self, context):
        prev_position = context["previous_state"]["position"]
        curr_position = context["current_state"]["position"]
        
        prev_distance = euclidean_distance(prev_position, self.target_position)
        curr_distance = euclidean_distance(curr_position, self.target_position)
        improvement = prev_distance - curr_distance
        
        return self.strength * improvement

class GoalReward(BaseReward):
    def __init__(self, target_position, threshold=0.1, reward=10.0):
        super().__init__()
        self.target_position = target_position
        self.threshold = threshold
        self.reward = reward
    
    def compute(self, context):
        curr_position = context["current_state"]["position"]
        distance = euclidean_distance(curr_position, self.target_position)
        return self.reward if distance < self.threshold else 0.0

class CollisionPenalty(BaseReward):
    def __init__(self, penalty=-5.0):
        super().__init__()
        self.penalty = penalty
    
    def compute(self, context):
        if detect_collision(context["current_state"]):
            return self.penalty
        return 0.0

# ... other components like EnergyEfficiencyReward, SmoothnessReward, etc.
Composing the Reward Function
Instead of a monolithic function, we compose these components:

Python
# Create and configure the reward engine
engine = RewardEngine()

# Add reward components
engine.add_component(DistanceReward(target_position=TARGET_POSITION, strength=1.5))
engine.add_component(GoalReward(target_position=TARGET_POSITION, threshold=0.1, reward=100.0))
engine.add_component(CollisionPenalty(penalty=-50.0))
engine.add_component(EnergyEfficiencyReward(penalty_factor=0.1))
engine.add_component(SmoothnessReward(penalty_factor=0.2))

# Set component weights (optional - can use defaults)
engine.set_weights({
    "DistanceReward": 1.0,
    "GoalReward": 1.0,
    "CollisionPenalty": 1.0,
    "EnergyEfficiencyReward": 1.0,
    "SmoothnessReward": 1.0
})
Computing Rewards with Transparency
Python
# Context object with state information
context = {
    "previous_state": previous_state,
    "current_state": current_state,
    "action": action,
    "done": done
}

# Compute the reward
total_reward = engine.compute(context)

# Get individual component contributions
contributions = engine.get_last_contributions()
print(f"Total reward: {total_reward}")
print(f"Component contributions: {contributions}")
Output:

Code
Total reward: 2.35
Component contributions: {
    'DistanceReward': 1.5,
    'GoalReward': 0.0,
    'CollisionPenalty': 0.0,
    'EnergyEfficiencyReward': -0.15,
    'SmoothnessReward': 1.0
}
Automated Reward Optimization
Instead of manual tuning, RLlama allows automatic optimization of component weights:

Python
from rllama.rewards.optimizer import RewardOptimizer

# Create an optimizer for the reward engine
optimizer = RewardOptimizer(engine)

# Define an evaluation function that tests the agent with given weights
def evaluate_weights(weights):
    engine.set_weights(weights)
    return run_evaluation_episodes(env, agent, n_episodes=10)

# Run optimization
best_weights = optimizer.optimize(evaluate_weights, n_trials=100)
print(f"Optimized weights: {best_weights}")

# Apply the optimized weights
engine.set_weights(best_weights)
Visualizing Reward Components
RLlama provides built-in visualization tools:

Python
from rllama.dashboard import RewardVisualizer

# Create a visualizer
visualizer = RewardVisualizer(engine)

# After training or evaluation
visualizer.plot_reward_history()
visualizer.plot_component_contributions()
visualizer.generate_report()
Adding New Components with Ease
Need a new reward component? Just add it without changing existing code:

Python
# Define a new component
class SafetyDistanceReward(BaseReward):
    def __init__(self, min_distance=1.0, penalty_factor=2.0):
        super().__init__()
        self.min_distance = min_distance
        self.penalty_factor = penalty_factor
    
    def compute(self, context):
        obstacles = context["current_state"]["obstacles"]
        agent_position = context["current_state"]["position"]
        
        min_dist = min([euclidean_distance(agent_position, obs) for obs in obstacles], default=float('inf'))
        
        if min_dist < self.min_distance:
            return -self.penalty_factor * (self.min_distance - min_dist)
        return 0.0

# Add it to the existing engine
engine.add_component(SafetyDistanceReward(min_distance=1.5, penalty_factor=3.0))
Integration with RL Frameworks
RLlama works seamlessly with existing RL frameworks:

Python
# With Gym environments
from rllama.integration import GymWrapper

gym_wrapper = GymWrapper(engine)
env = gym_wrapper.wrap(original_gym_env)

# Train with your favorite algorithm
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
Advantages of the RLlama Approach
Modularity: Each reward component is isolated and reusable
Transparency: Clear visibility into how each component contributes
Flexibility: Easy to add, remove, or modify components without affecting others
Optimization: Automated tuning of weights removes guesswork
Visualization: Built-in tools for understanding reward dynamics
Integration: Works with existing RL frameworks and environments
Maintainability: Clean separation of concerns makes code easier to maintain
Debugging: Quickly identify problematic components
In the following sections, we'll explore these benefits in greater detail and provide practical examples of RLlama in action. EOF

cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/installation.md << 'EOF'

Installation Guide
Basic Installation
RLlama is available on PyPI, making it easy to install with pip:

bash
pip install rllama
This installs the core functionality of RLlama, including the reward engine, basic reward components, and fundamental tools.

Installation with Optional Dependencies
RLlama offers several optional dependency groups for specific use cases:

Gym Integration
For working with OpenAI Gym environments:

bash
pip install "rllama[gym]"
This installs:

gym (≥0.17.0)
Stable Baselines Integration
For working with Stable Baselines3:

bash
pip install "rllama[sb3]"
This installs:

stable-baselines3 (≥1.5.0)
Visualization Tools
For advanced visualization and dashboard features:

bash
pip install "rllama[vis]"
This installs:

streamlit (≥1.10.0)
RLHF Support
For Reinforcement Learning from Human Feedback:

bash
pip install "rllama[rlhf]"
This installs:

tqdm (≥4.45.0)
Development Tools
For contributing to RLlama:

bash
pip install "rllama[dev]"
This installs:

pytest (≥6.0.0)
black (≥22.3.0)
isort (≥5.10.0)
build
twine
Complete Installation
To install all optional dependencies:

bash
pip install "rllama[all]"
Development Installation
For development work, install RLlama in editable mode:

bash
git clone https://github.com/ch33nchanyes/rllama.git
cd rllama
pip install -e ".[dev]"
This allows you to modify the code and see changes without reinstalling.

Requirements
RLlama requires:

Python ≥ 3.7
numpy ≥ 1.20.0
pyyaml ≥ 5.1
torch ≥ 1.9.0
matplotlib ≥ 3.3.0
optuna ≥ 3.0.0
Verifying Installation
After installation, verify that RLlama is installed correctly by running:

Python
import rllama
print(f"RLlama version: {rllama.__version__}")
This should print the installed version of RLlama without any errors. EOF

cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/quickstart.md << 'EOF'

Quickstart Guide
This guide will get you up and running with RLlama in minutes. We'll create a simple reward system and demonstrate the core functionality.

Basic Setup
First, let's import the necessary components:

Python
from rllama import RewardEngine
from rllama.rewards.components import (
    LengthReward,
    DiversityReward,
    ConstantReward
)
Creating a Reward Engine
The RewardEngine is the central component that manages all reward calculations:

Python
# Create a new reward engine
engine = RewardEngine()
Adding Reward Components
Next, let's add some reward components:

Python
# A reward based on text length (rewards responses close to 100 tokens)
engine.add_component(LengthReward(
    target_length=100,  # Target length in tokens
    strength=0.01,      # Scale factor for the reward
    mode="gaussian"     # Reward shape (gaussian = bell curve around target)
))

# A reward for response diversity compared to history
engine.add_component(DiversityReward(
    history_size=5,    # Number of past responses to consider
    strength=1.0       # Scale factor for the reward
))

# A constant baseline reward
engine.add_component(ConstantReward(value=0.5))
Computing Rewards
Now we can compute rewards for specific inputs:

Python
# Create a context object with input data
context = {
    "response": "This is a sample response that we want to evaluate.",
    "history": [
        "Previous response one.",
        "Another previous response.",
        "Yet another response in history."
    ]
}

# Compute the overall reward
reward = engine.compute(context)
print(f"Total reward: {reward}")

# Get individual component contributions
contributions = engine.get_last_contributions()
print("Component contributions:")
for component, value in contributions.items():
    print(f"  {component}: {value}")
Output:

Code
Total reward: 1.45
Component contributions:
  LengthReward: 0.27
  DiversityReward: 0.68
  ConstantReward: 0.5
Using Weights to Balance Components
You can adjust the relative importance of different components:

Python
# Set custom weights for each component
engine.set_weights({
    "LengthReward": 0.5,     # Reduce the importance of length
    "DiversityReward": 2.0,  # Increase the importance of diversity
    "ConstantReward": 1.0    # Keep constant reward as is
})

# Compute reward with new weights
reward = engine.compute(context)
print(f"Total reward with custom weights: {reward}")
print(f"Component contributions: {engine.get_last_contributions()}")
Output:

Code
Total reward with custom weights: 1.995
Component contributions: {'LengthReward': 0.135, 'DiversityReward': 1.36, 'ConstantReward': 0.5}
Configuring with YAML
For more complex setups, you can use YAML configuration files:

Python
# Create a YAML configuration
config = """
reward_components:
  - name: LengthReward
    params:
      target_length: 100
      strength: 0.01
      mode: gaussian
  - name: DiversityReward
    params:
      history_size: 5
      strength: 1.0
  - name: ConstantReward
    params:
      value: 0.5

weights:
  LengthReward: 0.5
  DiversityReward: 2.0
  ConstantReward: 1.0
"""

# Save to a file
with open("reward_config.yaml", "w") as f:
    f.write(config)

# Create engine from config file
engine_from_config = RewardEngine(config_path="reward_config.yaml")

# Same computation as before
reward = engine_from_config.compute(context)
print(f"Total reward from config: {reward}")
print(f"Component contributions: {engine_from_config.get_last_contributions()}")
Logging Rewards
Enable logging to track reward values over time:

Python
# Enable reward logging
engine.enable_logging(log_dir="./reward_logs")

# Compute rewards for multiple inputs
for i in range(10):
    context = {
        "response": f"This is sample response {i}",
        "history": previous_responses
    }
    reward = engine.compute(context)
    
    # Access logs programmatically
    logs = engine.get_logs()
Visualizing Rewards
RLlama provides tools for visualizing reward patterns:

Python
from rllama.dashboard import RewardVisualizer

# Create a visualizer for the reward engine
visualizer = RewardVisualizer(engine)

# Generate plots
visualizer.plot_reward_history()
visualizer.plot_component_contributions()
Next Steps
This quickstart covered the basics of RLlama. To learn more:

See the Core Concepts guide for deeper understanding
Check out the Usage Guide for more detailed examples
Explore the Reward Cookbook for common reward patterns
Try the Optimization Guide for automatic reward tuning EOF
cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/component-design.md << 'EOF'

Reward Component Design
This guide explains how to design custom reward components in RLlama, including best practices and advanced techniques.

Basic Component Structure
Every reward component in RLlama inherits from the BaseReward class:

Python
from rllama.rewards.base import BaseReward

class MyCustomReward(BaseReward):
    def __init__(self, param1=default1, param2=default2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def compute(self, context):
        # Extract relevant information from context
        # Perform reward calculation
        # Return a numerical reward value
        return reward_value
The Context Object
The context parameter passed to the compute method is a dictionary containing all the information needed to calculate rewards. Common keys include:

state: The current state of the environment
action: The action taken by the agent
next_state: The resulting state after taking the action
done: A boolean indicating if the episode is complete
response: For language models, the generated text
history: For sequential tasks, the history of previous states/responses
Your component should extract the information it needs from this context object.

Example: Simple Text Length Reward
Let's create a simple reward component that rewards text of a certain length:

Python
class TextLengthReward(BaseReward):
    def __init__(self, target_length=100, tolerance=20, max_reward=1.0):
        super().__init__()
        self.target_length = target_length
        self.tolerance = tolerance
        self.max_reward = max_reward
    
    def compute(self, context):
        # Extract the text from the context
        if "response" not in context:
            return 0.0  # No reward if response is missing
        
        text = context["response"]
        length = len(text.split())  # Simple word count
        
        # Calculate how close the length is to the target
        distance = abs(length - self.target_length)
        
        # If within tolerance, give proportional reward
        if distance <= self.tolerance:
            return self.max_reward * (1 - distance / self.tolerance)
        else:
            return 0.0
Best Practices for Reward Components
1. Single Responsibility
Each component should focus on a single aspect of behavior:

✅ Good: Separate components for text length, sentiment, and coherence.
❌ Bad: One component that handles all text quality aspects.

2. Robust to Missing Data
Components should handle cases where expected data is missing:

Python
def compute(self, context):
    # Check for required keys
    if "response" not in context:
        return 0.0  # Default reward when data is missing
    
    # Normal computation
    # ...
3. Configurable Parameters
Make important values configurable via constructor parameters:

Python
def __init__(self, strength=1.0, threshold=0.5, mode="linear"):
    super().__init__()
    self.strength = strength
    self.threshold = threshold
    self.mode = mode
4. Normalized Output Range
Keep reward values in a consistent range (typically -1 to 1 or 0 to 1) for easier balancing:

Python
def compute(self, context):
    # Calculate raw score
    raw_score = calculate_score(context)
    
    # Normalize to [0, 1] range
    normalized_score = max(0, min(raw_score, 1))
    
    return normalized_score * self.strength
5. Documentation
Thoroughly document what your component does and how it calculates rewards:

Python
class NoveltyReward(BaseReward):
    """Rewards novel states that haven't been seen before.
    
    Uses a memory buffer to track previously seen states and calculates
    novelty based on the minimum distance to any state in memory.
    
    Args:
        buffer_size: Maximum number of states to remember
        distance_fn: Function to calculate distance between states
        threshold: Minimum distance to be considered novel
        strength: Scaling factor for the reward
    """
    # ...
Advanced Component Design
State Tracking Between Calls
Some components need to track information across multiple calls:

Python
class TrendReward(BaseReward):
    def __init__(self, window_size=5, strength=1.0):
        super().__init__()
        self.window_size = window_size
        self.strength = strength
        self.values_history = []
    
    def compute(self, context):
        current_value = extract_value(context)
        self.values_history.append(current_value)
        
        # Keep history limited to window size
        if len(self.values_history) > self.window_size:
            self.values_history.pop(0)
            
        # Calculate trend
        if len(self.values_history) < 2:
            return 0.0
            
        trend = calculate_trend(self.values_history)
        return trend * self.strength
Component Composition
You can create meta-components that combine other components:

Python
class CompositeReward(BaseReward):
    def __init__(self):
        super().__init__()
        self.components = []
    
    def add_component(self, component, weight=1.0):
        self.components.append((component, weight))
    
    def compute(self, context):
        total = 0.0
        for component, weight in self.components:
            total += component.compute(context) * weight
        return total
Learning Components
Components can incorporate learning mechanisms:

Python
class AdaptiveReward(BaseReward):
    def __init__(self, initial_value=0.5, learning_rate=0.01):
        super().__init__()
        self.value = initial_value
        self.learning_rate = learning_rate
    
    def compute(self, context):
        reward = self.value
        
        # Update internal value based on feedback
        if "feedback" in context:
            error = context["feedback"] - self.value
            self.value += self.learning_rate * error
        
        return reward
Registering Custom Components
To make your components available in configuration files, register them:

Python
from rllama.rewards.registry import register_reward_component

# Register the component
register_reward_component(MyCustomReward)

# Now it can be used in YAML config
"""
reward_components:
  - name: MyCustomReward
    params:
      param1: value1
      param2: value2
"""
Testing Components
It's good practice to test reward components:

Python
def test_text_length_reward():
    # Create component
    reward = TextLengthReward(target_length=10, tolerance=5, max_reward=1.0)
    
    # Test ideal case
    context = {"response": "This is exactly ten words long period end now"}
    assert reward.compute(context) == 1.0
    
    # Test within tolerance
    context = {"response": "This is eight words long period"}
    assert 0 < reward.compute(context) < 1.0
    
    # Test outside tolerance
    context = {"response": "Too short"}
    assert reward.compute(context) == 0.0
Example Component Library
RLlama comes with many pre-built components:

LengthReward: Rewards based on text/response length
DiversityReward: Rewards diversity compared to history
CoherenceReward: Rewards logical coherence in responses
ProgressReward: Rewards making progress toward a goal
CuriosityReward: Rewards exploring new states
ConstraintReward: Enforces constraints on behavior
Check the API documentation for full details on all available components. EOF

cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/integration.md << 'EOF'

Integration with RL Frameworks
RLlama is designed to integrate seamlessly with popular reinforcement learning frameworks. This guide covers integration with:

OpenAI Gym environments
Stable Baselines3
Custom RL environments
OpenAI Gym Integration
RLlama provides a wrapper for Gym environments that intercepts and modifies rewards:

Python
from rllama import RewardEngine
from rllama.integration import GymWrapper
from rllama.rewards.components import ProgressReward, SparseSuccessReward

import gym

# Create a standard Gym environment
env = gym.make('CartPole-v1')

# Create and configure a reward engine
engine = RewardEngine()

# Add reward components
engine.add_component(ProgressReward(
    goal_pos=0.0,           # Target position
    goal_threshold=0.05,    # How close counts as "at goal"
    state_key=0             # Index of position in state vector
))

engine.add_component(SparseSuccessReward(
    reward_value=10.0,      # Bonus reward for task completion
))

# Create the wrapped environment
wrapped_env = GymWrapper(engine).wrap(env)

# Now use the wrapped environment with any RL algorithm
observation, info = wrapped_env.reset()

for _ in range(1000):
    action = wrapped_env.action_space.sample()  # your agent here
    observation, reward, terminated, truncated, info = wrapped_env.step(action)
    
    if terminated or truncated:
        observation, info = wrapped_env.reset()
Customizing the Context
You can customize how environment data is converted to the context object:

Python
def custom_context_builder(state, action, next_state, reward, done, info):
    context = {
        "state": state,
        "action": action, 
        "next_state": next_state,
        "original_reward": reward,
        "done": done,
        "info": info,
        # Add custom entries
        "position": next_state[0],
        "velocity": next_state[1],
        "distance_to_goal": abs(next_state[0])
    }
    return context

# Use the custom context builder with the wrapper
wrapped_env = GymWrapper(
    engine,
    context_builder=custom_context_builder
).wrap(env)
Reward Modes
RLlama supports different reward modification modes:

Python
# Replace the original reward entirely
wrapped_env = GymWrapper(
    engine,
    mode="replace"  # Default
).wrap(env)

# Add the RLlama reward to the original reward
wrapped_env = GymWrapper(
    engine,
    mode="add"
).wrap(env)

# Use the original reward and just log RLlama's reward
wrapped_env = GymWrapper(
    engine,
    mode="observe"
).wrap(env)
Stable Baselines3 Integration
RLlama integrates with Stable Baselines3 for easy training:

Python
from rllama.integration import StableBaselinesWrapper
from stable_baselines3 import PPO

# Create a wrapped environment as shown above
wrapped_env = GymWrapper(engine).wrap(env)

# Initialize a Stable Baselines model
model = PPO("MlpPolicy", wrapped_env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Save the trained model
model.save("ppo_cartpole_with_rllama")
Advanced Integration
For more advanced scenarios, RLlama provides a specialized wrapper:

Python
from rllama.integration import StableBaselinesWrapper

# Create the SB3 wrapper
sb3_wrapper = StableBaselinesWrapper(engine)

# Create a model with additional configurations
model = sb3_wrapper.create_model(
    algorithm="PPO",
    env=env,
    policy="MlpPolicy",
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)

# Train with callback for reward logging
model = sb3_wrapper.train(
    model,
    total_timesteps=100000,
    log_interval=100,
    log_dir="./sb3_logs"
)
Custom RL Environment Integration
For custom RL environments, you can use RLlama directly:

Python
class MyCustomEnvironment:
    def __init__(self):
        self.state = self.reset()
        self.reward_engine = RewardEngine()
        
        # Add reward components
        self.reward_engine.add_component(...)
        
    def reset(self):
        # Reset environment state
        self.state = initial_state
        return self.state
        
    def step(self, action):
        # Execute action and update state
        prev_state = self.state
        self.state = self.update_state(action)
        done = self.is_done()
        
        # Use RLlama to compute reward
        context = {
            "state": prev_state,
            "action": action,
            "next_state": self.state,
            "done": done
        }
        reward = self.reward_engine.compute(context)
        
        return self.state, reward, done, {}
Integrating with Deep Learning Frameworks
RLlama can also be integrated with deep learning frameworks like PyTorch:

Python
import torch
import torch.nn as nn
from rllama.models import MLPRewardModel

# Create a neural reward model
reward_model = MLPRewardModel(
    input_dim=state_dim + action_dim,
    hidden_dims=[128, 64],
    output_dim=1
)

# Add it to the reward engine
engine.add_component(NeuralReward(model=reward_model))

# Train the reward model
optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Get batch of state-action pairs and rewards
    states, actions, rewards = get_batch()
    
    # Forward pass
    inputs = torch.cat([states, actions], dim=1)
    predicted_rewards = reward_model(inputs).squeeze()
    
    # Calculate loss
    loss = nn.MSELoss()(predicted_rewards, rewards)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
Logging and Monitoring
RLlama provides tools for logging and monitoring rewards during training:

Python
# Enable logging in the reward engine
engine.enable_logging(
    log_dir="./reward_logs",
    log_frequency=100
)

# After training, analyze the logs
from rllama.dashboard import RewardVisualizer

visualizer = RewardVisualizer(engine)
visualizer.plot_reward_history()
visualizer.plot_component_contributions()
visualizer.generate_report()
Advanced Integration Examples
See the examples directory for more detailed integration examples, including:

Multi-task learning with RLlama
Hierarchical RL with reward shaping
Meta-learning for adaptive rewards 
