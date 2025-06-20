# RLlama: Composable Reward Engineering Framework

<p align="center">
  <img src="images/llamagym.jpg" alt="RLlama Logo" width="200"/>
</p>

<p align="center">
    <em>A composable reward engineering framework for reinforcement learning.</em>
</p>

## Overview

**RLlama** is a specialized Python framework designed to solve one of the most challenging problems in reinforcement learning: reward engineering. It provides a structured approach to creating, combining, and optimizing reward functions, making your RL systems more effective and easier to understand.

## Key Problems RLlama Solves

1. **Reward Function Complexity**: RL systems often need to balance multiple objectives, which leads to complex reward functions that are hard to design, debug, and maintain.

2. **Reward Hacking**: Poorly designed reward functions can lead to unintended agent behaviors as the agent finds loopholes to maximize rewards.

3. **Reward Sparsity**: Many real-world problems have sparse rewards, making learning difficult for agents.

4. **Transparency**: Understanding why an agent received a particular reward is often difficult with monolithic reward functions.

5. **Tuning Difficulty**: Adjusting reward functions through trial and error is time-consuming and inefficient.

## Core Features

- 🧩 **Modular Reward Components**: Mix and match reward functions to shape agent behavior
- 🔍 **Reward Optimization**: Automatically tune reward weights with Bayesian optimization
- 🧠 **Memory Systems**: Episodic and working memory for improved agent capabilities
- 📊 **Visualization Tools**: Track and analyze reward contributions
- 🔗 **RL Library Integration**: Seamless integration with OpenAI Gym and Stable Baselines3
- 💬 **RLHF Support**: Tools for Reinforcement Learning from Human Feedback
- 🌐 **Neural Network Reward Models**: Deep learning based reward modeling
- 🎛️ **Reward Normalization**: Multiple strategies for normalizing rewards

## Example

```python
from rllama import RewardEngine
from rllama.rewards.components import LengthReward, DiversityReward, CuriosityReward

# Create a reward engine
engine = RewardEngine()

# Add reward components
engine.add_component(LengthReward(target_length=100, strength=0.5))
engine.add_component(DiversityReward(history_size=10, strength=1.0))
engine.add_component(CuriosityReward(novelty_threshold=0.3))

# Set component weights
engine.set_weights({
    "LengthReward": 0.3,
    "DiversityReward": 0.5,
    "CuriosityReward": 0.2
})

# Compute rewards
context = {
    "response": "This is a test response", 
    "history": ["Previous response 1", "Previous response 2"],
    "state": current_state
}
reward = engine.compute(context)
print(f"Total reward: {reward}")
print(f"Component contributions: {engine.get_last_contributions()}")
Get Started
Ready to transform how you design reward functions?

Learn why RLlama is needed
See what RL is like without RLlama
See what RL is like with RLlama
Installation guide
Quickstart tutorial EOF
Code

Now let's create a file explaining why RLlama exists:

```bash
cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/why-rllama.md << 'EOF'
# Why RLlama Exists

## The Reward Engineering Problem

Reinforcement learning (RL) models learn to perform tasks by maximizing cumulative rewards. While this sounds straightforward in theory, in practice, designing effective reward functions is one of the most challenging aspects of RL.

### Traditional Reward Engineering Challenges

1. **The Specification Challenge**

   Translating human objectives into a mathematical reward function is inherently difficult. How do you quantify concepts like:
   
   - "Write a helpful response"
   - "Generate diverse but relevant content"
   - "Move naturally like a human"

2. **The Balancing Act**

   Most real-world tasks require balancing multiple objectives:
   
   - A robot needs to complete tasks quickly but safely
   - A language model needs to be helpful but not harmful
   - A game agent needs to score points while avoiding dangers

3. **Reward Hacking and Exploitation**

   RL agents often find unexpected ways to maximize rewards that don't align with the designer's intent:
   
   - A cleaning robot might hide dirt rather than clean it
   - A language model might repeat high-reward phrases excessively
   - A game agent might exploit glitches instead of playing as intended

4. **Transparency and Debugging**

   When an agent performs poorly, traditional reward functions offer limited insight:
   
   - Which part of the reward function is causing undesired behavior?
   - What specific actions lead to reward or penalty?
   - How can we systematically improve the reward function?

5. **Complex Reward Landscapes**

   Many tasks have sparse or deceptive reward landscapes:
   
   - Long-horizon tasks may have rewards only at completion
   - Intermediate steps might seem counter-productive
   - Local optima can trap agents in suboptimal behaviors

## The RLlama Solution

RLlama was created specifically to address these challenges through:

### 1. Composable Reward Design

RLlama allows you to build complex reward functions from simple, reusable components:

```python
engine = RewardEngine()
engine.add_component(TaskCompletionReward())
engine.add_component(SafetyConstraintReward())
engine.add_component(EnergyEfficiencyReward())
Each component can focus on a single aspect of the desired behavior, making the overall system more maintainable and understandable.

2. Transparent Reward Accounting
With RLlama, you can see exactly how much each component contributes to the total reward:

Python
reward = engine.compute(context)
contributions = engine.get_last_contributions()
# {'TaskCompletionReward': 0.8, 'SafetyConstraintReward': -0.1, 'EnergyEfficiencyReward': 0.3}
This transparency makes debugging and improvement much easier.

3. Automated Optimization
Instead of manually tuning reward weights, RLlama offers automated optimization:

Python
optimizer = RewardOptimizer(engine)
best_weights = optimizer.optimize(evaluation_function, n_trials=100)
engine.set_weights(best_weights)
This systematic approach finds effective reward configurations much faster than manual tuning.

4. Reward Shaping
RLlama includes tools for reward shaping to help agents learn in sparse reward environments:

Python
engine.add_shaping(ProgressShaping(distance_function=euclidean_distance))
5. Integration with Modern RL Systems
RLlama is designed to work seamlessly with popular RL frameworks:

Python
# With Gym environments
gym_wrapper = GymWrapper(engine)
env = gym_wrapper.wrap(original_gym_env)

# With Stable Baselines3
sb3_wrapper = StableBaselinesWrapper(engine)
model = sb3_wrapper.create_model(algorithm="PPO", env=env)
The Impact
By addressing these core challenges, RLlama allows researchers and developers to:

Iterate faster on reward function design
Understand better why agents behave as they do
Avoid common pitfalls like reward hacking
Build more complex agents that can balance multiple objectives
Create more robust reinforcement learning systems
In the following sections, we'll show concrete examples of RL with and without RLlama, demonstrating the practical advantages this framework provides. EOF

Code

Now let's create a file showing what RL is like without RLlama:

```bash
cat > /Users/cheencheen/Desktop/git/rl/rllama/docs/without-rllama.md << 'EOF'
# Reinforcement Learning Without RLlama

To understand the value RLlama provides, let's examine what reward engineering typically looks like without such a framework.

## Traditional Reward Function Implementation

In standard reinforcement learning implementations, reward functions are often implemented as monolithic functions that combine multiple objectives:

```python
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
In the next section, we'll see how RLlama transforms this process. 
