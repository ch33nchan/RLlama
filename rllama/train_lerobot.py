import numpy as np
import gymnasium as gym
import rllama
from rllama.envs.lerobot import LeRobotEnv
from rllama.utils.logger import Logger

# Register the environment
gym.register(
    id='LeRobot-v0',
    entry_point='rllama.envs.lerobot:LeRobotEnv',
    max_episode_steps=200,
    kwargs={'difficulty': 'easy'}
)

# Create environment
env = rllama.make_env("LeRobot-v0", seed=42)

# Create agent - SAC works well for continuous control tasks
agent = rllama.make_agent(
    "SAC",
    env=env,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    device="auto",
    verbose=True
)

# Set up experiment
experiment = rllama.Experiment(
    agent=agent,
    env=env,
    name="SAC-LeRobot",
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