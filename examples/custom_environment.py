import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

import rllama
from rllama.utils.logger import Logger

class SimpleGridWorld(gym.Env):
    """
    A simple grid world environment where the agent must navigate to a goal.
    
    State: (x, y) position on grid
    Actions: 0: up, 1: right, 2: down, 3: left
    Reward: -1 per step, +10 for reaching goal
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, grid_size=5, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: up, right, down, left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: (x, y) position
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, 
            shape=(2,), dtype=np.float32
        )
        
        # Set goal position
        self.goal_pos = np.array([grid_size-1, grid_size-1])
        
        # Reset the environment
        self.reset()
        
    def reset(self, seed=None, options=None):
        # Initialize the agent position
        self.agent_pos = np.array([0, 0])
        
        # Initialize step counter
        self.steps = 0
        
        # Return the initial observation
        return self.agent_pos.astype(np.float32), {}
    
    def step(self, action):
        # Increment step counter
        self.steps += 1
        
        # Move the agent
        if action == 0:  # up
            self.agent_pos[1] = min(self.grid_size-1, self.agent_pos[1]+1)
        elif action == 1:  # right
            self.agent_pos[0] = min(self.grid_size-1, self.agent_pos[0]+1)
        elif action == 2:  # down
            self.agent_pos[1] = max(0, self.agent_pos[1]-1)
        elif action == 3:  # left
            self.agent_pos[0] = max(0, self.agent_pos[0]-1)
        
        # Check if agent reached the goal
        done = np.array_equal(self.agent_pos, self.goal_pos)
        
        # Assign reward
        if done:
            reward = 10.0
        else:
            reward = -1.0
        
        # Return step information
        return self.agent_pos.astype(np.float32), reward, done, False, {}
    
    def render(self):
        if self.render_mode is None:
            return
            
        if self.render_mode == "human":
            # Print the grid
            grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            grid.fill('.')
            grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
            grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
            
            # Print the grid
            for row in reversed(grid):
                print(' '.join(row))
            print("\n")
            
        elif self.render_mode == "rgb_array":
            # Return a simple RGB representation
            grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            
            # Set goal position
            grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]
            
            # Set agent position
            grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]
            
            return grid
        
def main():
    """
    Demonstrates using a custom environment with RLlama.
    
    This example:
    1. Creates a custom grid world environment
    2. Registers it with Gymnasium
    3. Trains a DQN agent in the environment
    """
    # Register the custom environment
    gym.register(
        id='SimpleGridWorld-v0',
        entry_point='custom_environment:SimpleGridWorld',
        max_episode_steps=100,
    )
    
    # Create environment
    env = rllama.make_env("SimpleGridWorld-v0", seed=42)
    
    # Create agent
    agent = rllama.make_agent(
        "PPO",
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="auto",
        verbose=True
    )
    
    # Set up experiment
    experiment = rllama.Experiment(
        agent=agent,
        env=env,
        name="PPO-GridWorld",
        log_dir="logs"
    )
    
    # Train the agent
    experiment.train(total_steps=50000, log_interval=1000)
    
    # Evaluate the agent
    mean_reward, std_reward = experiment.evaluate(
        n_episodes=10,
        deterministic=True
    )
    print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Plot the learning curve
    logger = Logger("logs")
    logger.plot_metrics(
        ["episode_reward"],
        experiment_name="PPO-GridWorld",
        smoothing_window=3,
        show=True
    )

if __name__ == "__main__":
    main()