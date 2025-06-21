# Robotics Cookbook

Training robots involves balancing multiple, often conflicting, objectives: completing a task, ensuring safety, and conserving energy. RLlama is perfectly suited for this challenge.

## Recipe: Safe and Efficient Navigation

**Goal**: Train a mobile robot to navigate to a specific target location while avoiding obstacles and minimizing energy consumption.

### The Reward System

We'll create a reward system with three core components:

1.  **NavigationReward**: A dense reward that encourages the robot to make progress towards its goal on every step.
2.  **ObstacleAvoidanceReward**: A penalty-based component to ensure safety. It gives a large penalty for collisions and a smaller one for getting too close to objects.
3.  **EnergyEfficiencyReward**: A small penalty for every action taken to encourage the robot to find the shortest path.

### Full Implementation

```python
import numpy as np
from rllama.engine import RewardEngine
from rllama.rewards.base import BaseReward

# --- Define Custom Components for Robotics ---

class NavigationReward(BaseReward):
    """Rewards the agent for reducing its distance to a target."""
    def __init__(self, target_pos, strength=1.0):
        self.target_pos = np.array(target_pos)
        self.strength = strength
        self.previous_distance = None

    def compute(self, context):
        current_pos = np.array(context['state']['position'])
        distance = np.linalg.norm(current_pos - self.target_pos)

        if self.previous_distance is None:
            self.previous_distance = distance
            return 0.0
            
        reward = (self.previous_distance - distance) * self.strength
        self.previous_distance = distance
        
        # Reset on episode end
        if context.get('terminated') or context.get('truncated'):
            self.previous_distance = None

        return reward

class ObstacleAvoidanceReward(BaseReward):
    """Penalizes the agent for being too close to or hitting obstacles."""
    def __init__(self, collision_penalty=-50.0, proximity_penalty=-5.0, safe_distance=0.5):
        self.collision_penalty = collision_penalty
        self.proximity_penalty = proximity_penalty
        self.safe_distance = safe_distance

    def compute(self, context):
        if context['info'].get('collided', False):
            return self.collision_penalty
        
        agent_pos = np.array(context['state']['position'])
        min_dist_to_obstacle = float('inf')
        for obs_pos in context['state']['obstacle_positions']:
            dist = np.linalg.norm(agent_pos - np.array(obs_pos))
            if dist < min_dist_to_obstacle:
                min_dist_to_obstacle = dist
        
        if min_dist_to_obstacle < self.safe_distance:
            return self.proximity_penalty * (self.safe_distance - min_dist_to_obstacle)

        return 0.0

class EnergyEfficiencyReward(BaseReward):
    """Applies a small penalty for each action to encourage efficiency."""
    def __init__(self, penalty=-0.01):
        self.penalty = penalty
    
    def compute(self, context):
        return self.penalty

# --- Compose the Engine ---

# 1. Create the engine
robot_engine = RewardEngine()

# 2. Add components
robot_engine.add_component(NavigationReward(target_pos=[10, 10], strength=5.0))
robot_engine.add_component(ObstacleAvoidanceReward())
robot_engine.add_component(EnergyEfficiencyReward())

# 3. Set high-level weights
# We can decide that safety is the most important factor.
robot_engine.set_weights({
    "NavigationReward": 1.0,
    "ObstacleAvoidanceReward": 2.0, # Safety is twice as important as progress
    "EnergyEfficiencyReward": 1.0
})

# --- In the Training Loop ---

# context = {
#     "state": {"position": [x,y], "obstacle_positions": [[x1,y1], ...]},
#     "info": {"collided": False},
#     "terminated": False,
#     "truncated": False
# }
# final_reward = robot_engine.compute(context)