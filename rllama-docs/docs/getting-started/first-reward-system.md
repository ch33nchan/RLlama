---
id: first-reward-system
title: "Setting Up Your First Reward System"
sidebar_label: "First Reward System"
slug: /getting-started/first-reward-system
---

# Setting Up Your First Reward System

This guide takes you through the process of designing a complete reward system for a navigation task using RLlama.

## The Navigation Task

Imagine an agent navigating a 2D environment with obstacles, trying to reach a goal position. We want to create a reward system that encourages:

1. Moving toward the goal
2. Avoiding obstacles
3. Being energy-efficient (minimizing unnecessary movements)
4. Completing the task quickly

## Step 1: Design Reward Components

Let's design four reward components, one for each objective:

```python
import numpy as np
from rllama.rewards.base import BaseReward

class GoalDirectedReward(BaseReward):
    """Rewards the agent for making progress toward the goal."""
    
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = strength
        self.previous_distance = None
    
    def compute(self, context):
        agent_pos = context["state"]["position"]
        goal_pos = context["state"]["goal_position"]
        
        # Calculate distance to goal
        current_distance = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
        
        # First step, just store distance
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return 0.0
        
        # Calculate improvement in distance
        improvement = self.previous_distance - current_distance
        
        # Store for next time
        self.previous_distance = current_distance
        
        # Return reward (positive if getting closer, negative if moving away)
        return improvement * self.strength
    
    def reset(self):
        """Reset component state at the start of a new episode."""
        self.previous_distance = None

class ObstacleAvoidanceReward(BaseReward):
    """Penalizes the agent for getting too close to obstacles."""
    
    def __init__(self, safe_distance=1.0, penalty=2.0):
        super().__init__()
        self.safe_distance = safe_distance
        self.penalty = penalty
    
    def compute(self, context):
        agent_pos = context["state"]["position"]
        obstacles = context["state"]["obstacles"]
        
        # Calculate minimum distance to any obstacle
        distances = [np.linalg.norm(np.array(agent_pos) - np.array(obs_pos)) 
                    for obs_pos in obstacles]
        
        min_distance = min(distances) if distances else float('inf')
        
        # If too close to obstacle, apply penalty
        if min_distance < self.safe_distance:
            # Penalty increases as agent gets closer to obstacle
            return -self.penalty * (1.0 - min_distance / self.safe_distance)
        
        return 0.0  # No penalty if not too close

class EnergyEfficiencyReward(BaseReward):
    """Penalizes the agent for using excessive energy (making large or unnecessary movements)."""
    
    def __init__(self, efficiency_factor=0.1):
        super().__init__()
        self.efficiency_factor = efficiency_factor
    
    def compute(self, context):
        action = context["action"]
        
        # Calculate action magnitude (as a measure of energy use)
        action_magnitude = np.linalg.norm(np.array(action))
        
        # Penalize based on action magnitude
        return -action_magnitude * self.efficiency_factor

class TimeEfficiencyReward(BaseReward):
    """Encourages the agent to complete the task quickly."""
    
    def __init__(self, time_penalty=0.05):
        super().__init__()
        self.time_penalty = time_penalty
    
    def compute(self, context):
        # Simply return a fixed penalty for each step
        return -self.time_penalty
```

## Step 2: Combine Components into a Reward System

Now, let's create the reward engine and add our components:

```python
from rllama import RewardEngine

# Create reward engine
engine = RewardEngine()

# Add reward components
engine.add_component(GoalDirectedReward(strength=1.0))
engine.add_component(ObstacleAvoidanceReward(safe_distance=1.0, penalty=2.0))
engine.add_component(EnergyEfficiencyReward(efficiency_factor=0.1))
engine.add_component(TimeEfficiencyReward(time_penalty=0.05))

# Set component weights based on their relative importance
engine.set_weights({
    "GoalDirectedReward": 1.0,
    "ObstacleAvoidanceReward": 1.5,  # Safety is important
    "EnergyEfficiencyReward": 0.3,
    "TimeEfficiencyReward": 0.2
})
```

## Step 3: Create the Environment and Training Loop

For this example, we'll assume a simple custom environment. The integration with the reward system would look like this:

```python
# Create environment (custom or using Gym)
env = NavigationEnvironment()

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0
    
    # Reset stateful components
    for component in engine.components.values():
        if hasattr(component, 'reset'):
            component.reset()
    
    while not done:
        # Choose action (using your RL algorithm)
        action = agent.select_action(state)
        
        # Take action
        next_state, _, done, info = env.step(action)
        
        # Create context for reward calculation
        context = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "done": done,
            "info": info,
            "step": step
        }
        
        # Calculate reward using RLlama
        reward = engine.compute(context)
        
        # Learn from experience
        agent.update(state, action, reward, next_state, done)
        
        # Track statistics
        total_reward += reward
        state = next_state
        step += 1
        
        # Optional: Log component contributions
        if step % 10 == 0:
            contributions = engine.get_last_contributions()
            print(f"Step {step} contributions: {contributions}")
    
    print(f"Episode {episode}: Total reward = {total_reward}, Steps = {step}")
```

## Step 4: Analyze and Refine the Reward System

After training, analyze the component contributions to understand how the reward system is influencing the agent's behavior:

```python
import matplotlib.pyplot as plt

# Collect data during evaluation
states = []
rewards = []
contributions_history = []

state = env.reset()
done = False

# Reset stateful components
for component in engine.components.values():
    if hasattr(component, 'reset'):
        component.reset()

while not done:
    action = agent.select_action(state)
    next_state, _, done, info = env.step(action)
    
    context = {
        "state": state,
        "action": action,
        "next_state": next_state,
        "done": done,
        "info": info
    }
    
    reward = engine.compute(context)
    contributions = engine.get_last_contributions()
    
    states.append(state.copy())
    rewards.append(reward)
    contributions_history.append(contributions)
    
    state = next_state

# Plot the trajectory and reward components
plt.figure(figsize=(15, 5))

# Plot trajectory
plt.subplot(1, 2, 1)
positions = np.array([s["position"] for s in states])
plt.plot(positions[:, 0], positions[:, 1], 'b-')
goal = states[0]["goal_position"]
plt.plot(goal[0], goal[1], 'g*', markersize=15)
for obs in states[0]["obstacles"]:
    plt.plot(obs[0], obs[1], 'rx', markersize=10)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Agent Trajectory')

# Plot reward components
plt.subplot(1, 2, 2)
goal_rewards = [c.get("GoalDirectedReward", 0) for c in contributions_history]
obstacle_rewards = [c.get("ObstacleAvoidanceReward", 0) for c in contributions_history]
energy_rewards = [c.get("EnergyEfficiencyReward", 0) for c in contributions_history]
time_rewards = [c.get("TimeEfficiencyReward", 0) for c in contributions_history]

plt.plot(goal_rewards, label='Goal Progress')
plt.plot(obstacle_rewards, label='Obstacle Avoidance')
plt.plot(energy_rewards, label='Energy Efficiency')
plt.plot(time_rewards, label='Time Efficiency')
plt.plot(rewards, 'k--', label='Total Reward')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.title('Reward Component Contributions')

plt.tight_layout()
plt.show()
```

## Step 5: Fine-tune with RewardOptimizer

Once your system is basically working, you can use the `RewardOptimizer` to automatically find better component weights:

```python
from rllama.rewards.optimizer import RewardOptimizer

def evaluate_weights(weights):
    """Function to evaluate a set of reward weights."""
    engine.set_weights(weights)
    
    # Run multiple evaluation episodes
    total_score = 0
    num_episodes = 5
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Reset stateful components
        for component in engine.components.values():
            if hasattr(component, 'reset'):
                component.reset()
        
        while not done:
            action = agent.select_action(state)
            next_state, _, done, info = env.step(action)
            
            context = {
                "state": state,
                "action": action,
                "next_state": next_state,
                "done": done,
                "info": info
            }
            
            reward = engine.compute(context)
            episode_reward += reward
            state = next_state
        
        total_score += episode_reward
    
    return total_score / num_episodes

# Create optimizer
optimizer = RewardOptimizer(engine)

# Define search space for weights
search_space = {
    "GoalDirectedReward": (0.5, 2.0),
    "ObstacleAvoidanceReward": (1.0, 3.0),
    "EnergyEfficiencyReward": (0.1, 0.5),
    "TimeEfficiencyReward": (0.1, 0.5)
}

# Run optimization
best_weights = optimizer.optimize(
    evaluate_weights,
    n_trials=50,
    search_space=search_space
)

print(f"Best weights found: {best_weights}")

# Apply the optimized weights
engine.set_weights(best_weights)
```

## Summary

By following these steps, you've created a complete reward system for a navigation task that:

1. Breaks down the overall goal into specific behavioral objectives
2. Uses individual components to calculate rewards for each objective
3. Balances these components using weights
4. Provides transparency into what aspects of behavior are being rewarded
5. Can be automatically optimized

This modular, transparent approach is the core value proposition of RLlama, making reward engineering more systematic and effective.

For more advanced usage patterns, see the [Common Patterns](/docs/getting-started/common-patterns) guide.
