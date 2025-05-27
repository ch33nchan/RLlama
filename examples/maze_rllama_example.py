import gymnasium as gym
import numpy as np
from rllama.agent import RLlamaAgent
from rllama.rewards.base import BaseReward
from rllama.rewards.composer import RewardComposer
from rllama.memory import EpisodicMemory, WorkingMemory
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Custom Reward Components for Maze Solving
class ExplorationReward(BaseReward):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__(name, weight)
        self.visited_positions = set()
    
    def calculate(self, state, action):
        # Reward visiting new positions
        position = tuple(state['agent_pos']) if 'agent_pos' in state else tuple(state[:2])
        if position not in self.visited_positions:
            self.visited_positions.add(position)
            return self.weight * 1.0  # Bonus for new position
        return 0.0
    
    def reset(self):
        self.visited_positions.clear()

class EfficiencyReward(BaseReward):
    def __init__(self, name: str, weight: float = 1.0, max_steps: int = 1000):
        super().__init__(name, weight)
        self.max_steps = max_steps
        self.step_count = 0
    
    def calculate(self, state, action):
        self.step_count += 1
        # Penalize taking too many steps
        efficiency = max(0, (self.max_steps - self.step_count) / self.max_steps)
        return self.weight * efficiency * 0.01  # Small continuous reward
    
    def reset(self):
        self.step_count = 0

class GoalReward(BaseReward):
    def __init__(self, name: str, weight: float = 10.0):
        super().__init__(name, weight)
    
    def calculate(self, state, action):
        # Large reward for reaching goal
        if state.get('goal_reached', False):
            return self.weight
        return 0.0
    
    def reset(self):
        pass

class SafetyReward(BaseReward):
    def __init__(self, name: str, weight: float = 2.0):
        super().__init__(name, weight)
    
    def calculate(self, state, action):
        # Penalize hitting walls or dangerous areas
        if state.get('collision', False):
            return -self.weight
        return 0.0
    
    def reset(self):
        pass

# Custom Maze Environment Wrapper
class MazeEnvironmentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_pos = None
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Enhanced state information for RLlama
        enhanced_state = {
            'observation': obs,
            'agent_pos': getattr(self.env, 'agent_pos', obs[:2]),
            'goal_reached': reward > 0,  # Assuming positive reward means goal
            'collision': reward < 0,     # Assuming negative reward means collision
            'step_count': getattr(self.env, '_elapsed_steps', 0)
        }
        
        return enhanced_state, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        enhanced_state = {
            'observation': obs,
            'agent_pos': getattr(self.env, 'agent_pos', obs[:2]),
            'goal_reached': False,
            'collision': False,
            'step_count': 0
        }
        return enhanced_state, info

# Main Training Script
def train_maze_agent():
    # Create environment (using MiniGrid as example)
    try:
        import minigrid
        env = gym.make('MiniGrid-Empty-8x8-v0')
    except:
        # Fallback to a simple custom maze or CartPole
        env = gym.make('CartPole-v1')
    
    env = MazeEnvironmentWrapper(env)
    
    # Create RLlama reward components
    rewards = {
        "exploration": ExplorationReward(name="exploration", weight=0.5),
        "efficiency": EfficiencyReward(name="efficiency", weight=0.3),
        "goal": GoalReward(name="goal", weight=10.0),
        "safety": SafetyReward(name="safety", weight=2.0)
    }
    
    # Create reward composer
    composer = RewardComposer(rewards, composition_strategy="weighted_sum")
    
    # Create RLlama agent with memory
    episodic_memory = EpisodicMemory(capacity=1000)
    working_memory = WorkingMemory(capacity=50)
    
    agent = RLlamaAgent(
        episodic_memory=episodic_memory,
        working_memory=working_memory,
        reward_composer=composer
    )
    
    # Custom reward function for the RL algorithm
    def rllama_reward_function(state, action, next_state, done, info):
        # Calculate RLlama composite reward
        rllama_reward = composer.calculate(next_state, action)
        
        # Store experience in memory
        experience = {
            'state': state,
            'action': action,
            'reward': rllama_reward,
            'next_state': next_state,
            'done': done
        }
        agent.store_experience(experience)
        
        return rllama_reward
    
    # Training loop with PPO
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=3e-4)
    
    # Custom callback to use RLlama rewards
    class RLlamaCallback:
        def __init__(self, agent, composer):
            self.agent = agent
            self.composer = composer
            self.episode_rewards = []
        
        def on_step(self, locals_, globals_):
            # Get current state and action
            state = locals_.get('obs', {})
            action = locals_.get('actions', 0)
            
            # Calculate RLlama reward
            rllama_reward = self.composer.calculate(state, action)
            
            # Override the environment reward
            locals_['rewards'] = np.array([rllama_reward])
            
            return True
    
    # Train the model
    print("Training maze agent with RLlama rewards...")
    model.learn(total_timesteps=50000)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate RLlama reward for evaluation
        rllama_reward = composer.calculate(obs, action)
        total_reward += rllama_reward
        
        if terminated or truncated:
            break
    
    print(f"Total RLlama reward: {total_reward}")
    print(f"Steps taken: {step + 1}")
    
    # Reset all reward components for next episode
    for reward_component in rewards.values():
        reward_component.reset()

if __name__ == "__main__":
    train_maze_agent()