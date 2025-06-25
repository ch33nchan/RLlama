#!/usr/bin/env python3
"""
RLlama Cookbook: Gym + Stable Baselines3 Integration

This cookbook demonstrates how to integrate RLlama reward engineering
with OpenAI Gym environments and Stable Baselines3 algorithms.

We'll cover:
1. FrozenLake environment with custom reward shaping
2. CartPole with progress and stability rewards
3. Atari Pong with multi-objective rewards
4. Custom reward wrappers for any Gym environment
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

# Add RLlama to path
sys.path.append(os.path.abspath("../.."))

# Stable Baselines3 imports
try:
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  Stable Baselines3 not available. Install with: pip install stable-baselines3[extra]")
    SB3_AVAILABLE = False

# RLlama imports
from rllama import RewardEngine, BaseReward, register_reward_component
from rllama.rewards.components.common import ProgressReward
from rllama.rewards.components.specific_rewards import DiversityReward, CuriosityReward

# ============================================================================
# Custom Reward Components for RL
# ============================================================================

@register_reward_component
class StabilityReward(BaseReward):
    """Rewards stable behavior (low action variance)."""
    
    def __init__(self, window_size=10, strength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.strength = strength
        self.action_history = []
        
    def calculate(self, context):
        action = context.get('action', 0)
        self.action_history.append(action)
        
        if len(self.action_history) > self.window_size:
            self.action_history.pop(0)
            
        if len(self.action_history) < 2:
            return 0.0
            
        # Reward low variance in actions
        variance = np.var(self.action_history)
        stability_reward = self.strength / (1.0 + variance)
        
        return stability_reward

@register_reward_component
class ExplorationReward(BaseReward):
    """Rewards visiting new states."""
    
    def __init__(self, grid_size=10, strength=0.05, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.strength = strength
        self.visited_states = set()
        
    def calculate(self, context):
        observation = context.get('observation', [0, 0])
        
        # Discretize continuous observations
        if isinstance(observation, (list, np.ndarray)):
            if len(observation) >= 2:
                # For CartPole-like environments
                x_pos = int(observation[0] * self.grid_size)
                x_vel = int(observation[1] * self.grid_size)
                state_key = (x_pos, x_vel)
            else:
                state_key = tuple(int(x * self.grid_size) for x in observation)
        else:
            state_key = int(observation)
            
        if state_key not in self.visited_states:
            self.visited_states.add(state_key)
            return self.strength
        else:
            return 0.0

@register_reward_component
class DistanceReward(BaseReward):
    """Rewards getting closer to a target position."""
    
    def __init__(self, target_position=None, strength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.target_position = target_position or [0.0, 0.0]
        self.strength = strength
        self.previous_distance = None
        
    def calculate(self, context):
        observation = context.get('observation', [0, 0])
        
        if isinstance(observation, (list, np.ndarray)) and len(observation) >= 2:
            current_pos = observation[:2]
        else:
            current_pos = [float(observation), 0.0]
            
        # Calculate distance to target
        distance = np.linalg.norm(np.array(current_pos) - np.array(self.target_position))
        
        if self.previous_distance is not None:
            # Reward for getting closer
            progress = self.previous_distance - distance
            reward = self.strength * progress
        else:
            reward = 0.0
            
        self.previous_distance = distance
        return reward

# ============================================================================
# Gym Environment Wrapper with RLlama
# ============================================================================

class RLlamaRewardWrapper(gym.Wrapper):
    """
    Gym wrapper that integrates RLlama reward engineering.
    Replaces or augments the original environment reward.
    """
    
    def __init__(self, env, config_path, replace_reward=False, reward_scaling=1.0):
        """
        Initialize the RLlama reward wrapper.
        
        Args:
            env: Gym environment to wrap
            config_path: Path to RLlama configuration file
            replace_reward: If True, replace env reward; if False, add to it
            reward_scaling: Scaling factor for RLlama rewards
        """
        super().__init__(env)
        self.replace_reward = replace_reward
        self.reward_scaling = reward_scaling
        
        # Initialize RLlama engine
        self.reward_engine = RewardEngine(config_path, verbose=False)
        
        # Track episode statistics
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        
    def reset(self, **kwargs):
        """Reset environment and RLlama state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_rewards = []
        self.episode_actions = []
        
        return obs, info
        
    def step(self, action):
        """Step environment and compute RLlama reward."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Create context for RLlama
        context = {
            'observation': obs,
            'action': action,
            'original_reward': reward,
            'step': self.episode_step,
            'terminated': terminated,
            'truncated': truncated,
            'episode_actions': self.episode_actions.copy(),
            'episode_rewards': self.episode_rewards.copy()
        }
        
        # Compute RLlama reward
        rllama_reward = self.reward_engine.compute(context)
        scaled_rllama_reward = rllama_reward * self.reward_scaling
        
        # Combine rewards
        if self.replace_reward:
            final_reward = scaled_rllama_reward
        else:
            final_reward = reward + scaled_rllama_reward
            
        # Update tracking
        self.episode_step += 1
        self.episode_actions.append(action)
        self.episode_rewards.append(final_reward)
        
        # Add RLlama info
        info['rllama_reward'] = scaled_rllama_reward
        info['original_reward'] = reward
        info['final_reward'] = final_reward
        
        return obs, final_reward, terminated, truncated, info

# ============================================================================
# Cookbook Examples
# ============================================================================

def create_config_file(config_path, config_type="cartpole"):
    """Create RLlama configuration files for different environments."""
    
    configs = {
        "frozenlake": {
            "reward_components": [
                {
                    "name": "ExplorationReward",
                    "params": {"strength": 0.1, "grid_size": 4}
                },
                {
                    "name": "ProgressReward", 
                    "params": {"target_key": "observation", "strength": 0.05}
                }
            ],
            "shaping_config": {
                "ExplorationReward": 1.0,
                "ProgressReward": 0.5
            }
        },
        
        "cartpole": {
            "reward_components": [
                {
                    "name": "StabilityReward",
                    "params": {"window_size": 10, "strength": 0.1}
                },
                {
                    "name": "DistanceReward",
                    "params": {"target_position": [0.0, 0.0], "strength": 0.05}
                },
                {
                    "name": "ExplorationReward",
                    "params": {"strength": 0.02, "grid_size": 20}
                }
            ],
            "shaping_config": {
                "StabilityReward": 0.3,
                "DistanceReward": 0.5, 
                "ExplorationReward": 0.2
            }
        },
        
        "atari": {
            "reward_components": [
                {
                    "name": "DiversityReward",
                    "params": {"history_size": 20, "strength": 0.1}
                },
                {
                    "name": "CuriosityReward", 
                    "params": {"scaling": 0.05}
                },
                {
                    "name": "ProgressReward",
                    "params": {"strength": 0.1}
                }
            ],
            "shaping_config": {
                "DiversityReward": 0.4,
                "CuriosityReward": 0.3,
                "ProgressReward": 0.3
            }
        }
    }
    
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(configs[config_type], f, default_flow_style=False)

class RLlamaCallback(BaseCallback):
    """Stable Baselines3 callback for RLlama integration."""
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rllama_rewards = []
        self.original_rewards = []
        
    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Get info from the environment
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'rllama_reward' in info:
                self.rllama_rewards.append(info['rllama_reward'])
                self.original_rewards.append(info['original_reward'])
                
        # Log statistics periodically
        if len(self.rllama_rewards) > 0 and len(self.rllama_rewards) % self.log_freq == 0:
            avg_rllama = np.mean(self.rllama_rewards[-self.log_freq:])
            avg_original = np.mean(self.original_rewards[-self.log_freq:])
            
            self.logger.record('rllama/avg_reward', avg_rllama)
            self.logger.record('rllama/avg_original_reward', avg_original)
            
        return True

def example_1_frozenlake():
    """Example 1: FrozenLake with exploration rewards."""
    print("\n" + "="*60)
    print("🧊 Example 1: FrozenLake with Exploration Rewards")
    print("="*60)
    
    # Create config
    config_path = "./frozenlake_config.yaml"
    create_config_file(config_path, "frozenlake")
    
    # Create environment
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode=None)
    wrapped_env = RLlamaRewardWrapper(env, config_path, replace_reward=False, reward_scaling=1.0)
    
    print(f"Environment: {env.spec.id}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test the environment
    print("\nTesting RLlama integration...")
    obs, info = wrapped_env.reset()
    total_reward = 0
    total_rllama_reward = 0
    
    for step in range(20):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        total_reward += reward
        total_rllama_reward += info['rllama_reward']
        
        print(f"Step {step+1}: Action={action}, Obs={obs}, "
              f"Original={info['original_reward']:.3f}, "
              f"RLlama={info['rllama_reward']:.3f}, "
              f"Final={info['final_reward']:.3f}")
        
        if terminated or truncated:
            break
            
    print(f"\nTotal Original Reward: {total_reward - total_rllama_reward:.3f}")
    print(f"Total RLlama Reward: {total_rllama_reward:.3f}")
    print(f"Total Final Reward: {total_reward:.3f}")
    
    wrapped_env.close()

def example_2_cartpole_with_sb3():
    """Example 2: CartPole with Stable Baselines3 training."""
    print("\n" + "="*60)
    print("🎯 Example 2: CartPole with Stable Baselines3")
    print("="*60)
    
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 not available. Skipping this example.")
        return
        
    # Create config
    config_path = "./cartpole_config.yaml"
    create_config_file(config_path, "cartpole")
    
    # Create environment function
    def make_env():
        env = gym.make('CartPole-v1')
        env = RLlamaRewardWrapper(env, config_path, replace_reward=False, reward_scaling=0.1)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    print("Training PPO agent with RLlama rewards...")
    
    # Create callback
    callback = RLlamaCallback(log_freq=500)
    
    # Create and train model
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./cartpole_tensorboard/")
    model.learn(total_timesteps=5000, callback=callback)
    
    print("Training completed!")
    
    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    total_rewards = []
    
    for episode in range(5):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            if done[0]:
                break
                
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
        
    print(f"Average reward over 5 episodes: {np.mean(total_rewards):.2f}")
    
    env.close()

def example_3_atari_pong():
    """Example 3: Atari Pong with multi-objective rewards."""
    print("\n" + "="*60)
    print("🏓 Example 3: Atari Pong with Multi-Objective Rewards")
    print("="*60)
    
    try:
        # Try to create Atari environment
        env = gym.make('ALE/Pong-v5', render_mode=None)
        print(f"Environment: {env.spec.id}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Create config
        config_path = "./atari_config.yaml"
        create_config_file(config_path, "atari")
        
        # Wrap environment
        wrapped_env = RLlamaRewardWrapper(env, config_path, replace_reward=False, reward_scaling=0.01)
        
        print("\nTesting Atari integration...")
        obs, info = wrapped_env.reset()
        total_reward = 0
        
        for step in range(100):  # Short test
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            
            total_reward += reward
            
            if step % 20 == 0:
                print(f"Step {step}: Original={info['original_reward']:.3f}, "
                      f"RLlama={info['rllama_reward']:.3f}, "
                      f"Final={info['final_reward']:.3f}")
            
            if terminated or truncated:
                break
                
        print(f"\nTotal reward after {step+1} steps: {total_reward:.3f}")
        wrapped_env.close()
        
    except gym.error.UnregisteredEnv:
        print("❌ Atari environments not available. Install with: pip install gymnasium[atari]")
    except Exception as e:
        print(f"❌ Error with Atari environment: {e}")

def example_4_custom_reward_analysis():
    """Example 4: Analyze reward components in detail."""
    print("\n" + "="*60)
    print("📊 Example 4: Custom Reward Analysis")
    print("="*60)
    
    # Create config
    config_path = "./analysis_config.yaml"
    create_config_file(config_path, "cartpole")
    
    # Create environment
    env = gym.make('CartPole-v1')
    wrapped_env = RLlamaRewardWrapper(env, config_path, replace_reward=False, reward_scaling=1.0)
    
    # Collect detailed reward data
    print("Collecting reward component data...")
    
    reward_data = {
        'steps': [],
        'original_rewards': [],
        'stability_rewards': [],
        'distance_rewards': [],
        'exploration_rewards': [],
        'total_rllama_rewards': []
    }
    
    obs, info = wrapped_env.reset()
    
    for step in range(200):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        # Get component rewards from the engine
        context = {
            'observation': obs,
            'action': action,
            'step': step
        }
        
        component_rewards = wrapped_env.reward_engine.composer.calculate(context)
        
        # Store data
        reward_data['steps'].append(step)
        reward_data['original_rewards'].append(info['original_reward'])
        reward_data['stability_rewards'].append(component_rewards.get('StabilityReward', 0))
        reward_data['distance_rewards'].append(component_rewards.get('DistanceReward', 0))
        reward_data['exploration_rewards'].append(component_rewards.get('ExplorationReward', 0))
        reward_data['total_rllama_rewards'].append(info['rllama_reward'])
        
        if terminated or truncated:
            obs, info = wrapped_env.reset()
            
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Individual reward components
    plt.subplot(2, 2, 1)
    plt.plot(reward_data['steps'], reward_data['stability_rewards'], label='Stability', alpha=0.7)
    plt.plot(reward_data['steps'], reward_data['distance_rewards'], label='Distance', alpha=0.7)
    plt.plot(reward_data['steps'], reward_data['exploration_rewards'], label='Exploration', alpha=0.7)
    plt.title('Individual Reward Components')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Total rewards comparison
    plt.subplot(2, 2, 2)
    plt.plot(reward_data['steps'], reward_data['original_rewards'], label='Original', alpha=0.7)
    plt.plot(reward_data['steps'], reward_data['total_rllama_rewards'], label='RLlama', alpha=0.7)
    plt.title('Original vs RLlama Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative rewards
    plt.subplot(2, 2, 3)
    cumulative_original = np.cumsum(reward_data['original_rewards'])
    cumulative_rllama = np.cumsum(reward_data['total_rllama_rewards'])
    plt.plot(reward_data['steps'], cumulative_original, label='Cumulative Original', alpha=0.7)
    plt.plot(reward_data['steps'], cumulative_rllama, label='Cumulative RLlama', alpha=0.7)
    plt.title('Cumulative Rewards')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution
    plt.subplot(2, 2, 4)
    plt.hist(reward_data['original_rewards'], alpha=0.5, label='Original', bins=20)
    plt.hist(reward_data['total_rllama_rewards'], alpha=0.5, label='RLlama', bins=20)
    plt.title('Reward Distribution')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./reward_analysis.png', dpi=150, bbox_inches='tight')
    print("📊 Reward analysis plot saved as 'reward_analysis.png'")
    
    # Print statistics
    print(f"\nReward Statistics:")
    print(f"Original Reward - Mean: {np.mean(reward_data['original_rewards']):.3f}, "
          f"Std: {np.std(reward_data['original_rewards']):.3f}")
    print(f"RLlama Reward - Mean: {np.mean(reward_data['total_rllama_rewards']):.3f}, "
          f"Std: {np.std(reward_data['total_rllama_rewards']):.3f}")
    print(f"Stability Component - Mean: {np.mean(reward_data['stability_rewards']):.3f}")
    print(f"Distance Component - Mean: {np.mean(reward_data['distance_rewards']):.3f}")
    print(f"Exploration Component - Mean: {np.mean(reward_data['exploration_rewards']):.3f}")
    
    wrapped_env.close()

def main():
    """Run all cookbook examples."""
    print("🦙 RLlama x Gym x Stable Baselines3 Cookbook")
    print("=" * 60)
    print("This cookbook demonstrates RLlama integration with:")
    print("• OpenAI Gym environments")
    print("• Stable Baselines3 algorithms") 
    print("• Custom reward engineering")
    print("• Multi-objective reward optimization")
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    try:
        # Run examples
        example_1_frozenlake()
        example_2_cartpole_with_sb3()
        example_3_atari_pong()
        example_4_custom_reward_analysis()
        
        print("\n" + "="*60)
        print("✅ All cookbook examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("• Experiment with different reward component combinations")
        print("• Try the reward optimizer for hyperparameter tuning")
        print("• Implement your own custom reward components")
        print("• Scale up to more complex environments")
        
    except Exception as e:
        print(f"\n❌ Error running cookbook: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
