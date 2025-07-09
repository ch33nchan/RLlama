import sys
import os
import time
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import torch

import rllama


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm name")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", type=str, default="human", help="Render mode")
    args = parser.parse_args()
    
    # Create environment with rendering
    env = rllama.make_env(args.env, render_mode=args.render)
    
    # Create agent
    agent = rllama.make_agent(args.algorithm, env=env)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    agent.load_state_dict(checkpoint["agent"])
    
    # Run episodes
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(obs, evaluate=True)
            
            # Take step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Add small delay for visualization
            time.sleep(0.01)
            
        print(f"Episode {episode+1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()


if __name__ == "__main__":
    main()