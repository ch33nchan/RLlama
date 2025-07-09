"""
Train an RLlama agent on a LeRobot environment.
"""
import os
import sys
import argparse
import numpy as np
import gymnasium as gym

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

import rllama
from rllama.envs.lerobot_hf import make_lerobot_env, list_lerobot_envs, is_using_stub
from rllama.utils.logger import Logger

def main(args):
    """
    Train an RLlama agent on a LeRobot environment.
    
    Args:
        args: Command line arguments
    """
    # Check if we're using the stub
    if is_using_stub():
        print("NOTICE: Using stub implementation of LeRobot (real package requires Python 3.10+)")
    else:
        print("Using official LeRobot package")
    
    # List available environments if requested
    if args.list_envs:
        print("\nAvailable LeRobot environments:")
        try:
            envs = list_lerobot_envs()
            for i, env_id in enumerate(envs):
                print(f"{i+1}. {env_id}")
        except Exception as e:
            print(f"Error listing environments: {e}")
        return
    
    # Create LeRobot environment
    env = make_lerobot_env(
        env_id=args.env_id,
        render_mode=args.render_mode if args.render else None,
        max_episode_steps=args.max_steps
    )
    
    # Create agent based on action space type
    if isinstance(env.action_space, gym.spaces.Discrete):
        agent_type = args.discrete_agent
    else:
        agent_type = args.continuous_agent
        
    print(f"Using agent type: {agent_type} for environment: {args.env_id}")
    
    # Create agent with appropriate parameters
    if agent_type == "PPO":
        agent = rllama.make_agent(
            agent_type,
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            device=args.device,
            verbose=True
        )
    elif agent_type in ["DDPG", "TD3", "SAC"]:
        agent = rllama.make_agent(
            agent_type,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=0.005,
            device=args.device,
            verbose=True
        )
    elif agent_type == "DQN":
        agent = rllama.make_agent(
            agent_type,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_freq=500,
            device=args.device,
            verbose=True
        )
    else:
        agent = rllama.make_agent(
            agent_type,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            device=args.device,
            verbose=True
        )
    
    # Set up experiment
    experiment = rllama.Experiment(
        agent=agent,
        env=env,
        name=f"{agent_type}-{args.env_id}",
        log_dir=args.log_dir
    )
    
    # Train the agent
    experiment.train(total_steps=args.total_steps, log_interval=args.log_interval)
    
    # Evaluate the agent
    if args.evaluate:
        mean_reward, std_reward = experiment.evaluate(
            n_episodes=args.eval_episodes,
            deterministic=True
        )
        print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save the trained agent if requested
    if args.save_model:
        save_path = os.path.join(args.log_dir, f"{agent_type}_{args.env_id}.pt")
        experiment.save(save_path)
        print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RLlama agent on LeRobot environment")
    
    # Environment options
    parser.add_argument("--env_id", type=str, default="ReachTarget-v0", 
                        help="LeRobot environment ID")
    parser.add_argument("--list_envs", action="store_true", 
                        help="List available LeRobot environments and exit")
    parser.add_argument("--max_steps", type=int, default=1000, 
                        help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during training")
    parser.add_argument("--render_mode", type=str, default="human", 
                        choices=["human", "rgb_array"], 
                        help="Rendering mode")
    
    # Agent options
    parser.add_argument("--discrete_agent", type=str, default="DQN", 
                        choices=["DQN", "A2C", "PPO"], 
                        help="Agent type for discrete action spaces")
    parser.add_argument("--continuous_agent", type=str, default="SAC", 
                        choices=["DDPG", "TD3", "SAC", "PPO"], 
                        help="Agent type for continuous action spaces")
    parser.add_argument("--learning_rate", type=float, default=3e-4, 
                        help="Learning rate")
    parser.add_argument("--buffer_size", type=int, default=100000, 
                        help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size")
    parser.add_argument("--n_steps", type=int, default=128,
                        help="Number of steps for PPO")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to run on (cpu, cuda, auto)")
    
    # Training options
    parser.add_argument("--total_steps", type=int, default=100000, 
                        help="Total training steps")
    parser.add_argument("--log_interval", type=int, default=1000, 
                        help="Logging interval")
    parser.add_argument("--log_dir", type=str, default="logs", 
                        help="Directory to save logs")
    parser.add_argument("--save_model", action="store_true", 
                        help="Save the trained model")
    
    # Evaluation options
    parser.add_argument("--evaluate", action="store_true", 
                        help="Evaluate the agent after training")
    parser.add_argument("--eval_episodes", type=int, default=10, 
                        help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    main(args)
