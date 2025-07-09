import numpy as np
import gymnasium as gym
import os
import sys

# Add the parent directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from rllama.envs
from rllama.envs.lerobot import LeRobotEnv

def test_lerobot():
    """Test the LeRobot environment functionality."""
    print("Creating LeRobot environment...")
    env = LeRobotEnv(max_steps=100, difficulty="easy", render_mode="human")
    
    print("\nTesting environment reset...")
    obs, info = env.reset(seed=42)
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Initial position: {obs['position']}")
    print(f"Target position: {obs['target_position']}")
    print(f"Target relative: {obs['target_relative']}")
    
    total_reward = 0
    print("\nTesting environment steps...")
    for i in range(10):
        action = env.action_space.sample()
        print(f"\nStep {i+1}")
        print(f"Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Reward: {reward:.2f}")
        print(f"New position: {obs['position']}")
        print(f"Distance to target: {np.linalg.norm(obs['target_relative']):.2f}")
        print(f"Done: {terminated or truncated}")
        
        env.render()
        
        if terminated or truncated:
            print("Episode finished early")
            break
    
    print(f"\nTotal reward after 10 steps: {total_reward:.2f}")
    env.close()
    print("\nLeRobot environment test completed successfully!")

if __name__ == "__main__":
    test_lerobot()