from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rllama.core.experience import Experience


class Runner:
    """
    Runner for collecting experience from environment interactions.
    
    Attributes:
        env: Environment to collect experience from
        agent: Agent to use for selecting actions
        device: Device to run computations on
    """
    
    def __init__(
        self,
        env: Any,
        agent: Any,
        device: str = "cpu",
    ):
        """
        Initialize a runner.
        
        Args:
            env: Environment to collect experience from
            agent: Agent to use for selecting actions
            device: Device to run computations on
        """
        self.env = env
        self.agent = agent
        self.device = device
        
        # Initialize state
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        self.episodes_completed = 0
        
    def reset(self) -> None:
        """Reset the runner state."""
        self.current_obs, _ = self.env.reset()
        self.episode_reward = 0
        self.episode_steps = 0
        
    def collect_one_step(
        self, 
        deterministic: bool = False
    ) -> Tuple[Experience, Dict[str, Any]]:
        """
        Collect a single step of experience.
        
        Args:
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (experience, info dictionary)
        """
        # Reset if needed
        if self.current_obs is None:
            self.reset()
            
        # Select action
        action = self.agent.select_action(self.current_obs, evaluate=deterministic)
        
        # Take a step in the environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Create experience
        experience = Experience(
            obs=self.current_obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            info=info
        )
        
        # Update state
        self.current_obs = next_obs
        self.episode_reward += reward
        self.episode_steps += 1
        
        # Handle episode termination
        step_info = {"terminated": False}
        if done:
            self.episodes_completed += 1
            step_info = {
                "terminated": True,
                "episode_reward": self.episode_reward,
                "episode_length": self.episode_steps,
                "episodes_completed": self.episodes_completed
            }
            self.reset()
            
        return experience, step_info
    
    def collect_rollout(
        self, 
        n_steps: int, 
        deterministic: bool = False
    ) -> Tuple[List[Experience], Dict[str, Any]]:
        """
        Collect a rollout of experiences.
        
        Args:
            n_steps: Number of steps to collect
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (experiences, info dictionary)
        """
        experiences = []
        episode_rewards = []
        episode_lengths = []
        
        # Reset if needed
        if self.current_obs is None:
            self.reset()
            
        for _ in range(n_steps):
            experience, step_info = self.collect_one_step(deterministic)
            experiences.append(experience)
            
            if step_info["terminated"]:
                episode_rewards.append(step_info["episode_reward"])
                episode_lengths.append(step_info["episode_length"])
                
        # Compute rollout statistics
        rollout_info = {
            "episodes_completed": len(episode_rewards),
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
            "experiences_collected": len(experiences)
        }
        
        return experiences, rollout_info
    
    def collect_episodes(
        self, 
        n_episodes: int, 
        deterministic: bool = False
    ) -> Tuple[List[Experience], Dict[str, Any]]:
        """
        Collect a specified number of complete episodes.
        
        Args:
            n_episodes: Number of episodes to collect
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (experiences, info dictionary)
        """
        episodes_completed = 0
        experiences = []
        episode_rewards = []
        episode_lengths = []
        
        # Reset if needed
        if self.current_obs is None:
            self.reset()
            
        while episodes_completed < n_episodes:
            experience, step_info = self.collect_one_step(deterministic)
            experiences.append(experience)
            
            if step_info["terminated"]:
                episodes_completed += 1
                episode_rewards.append(step_info["episode_reward"])
                episode_lengths.append(step_info["episode_length"])
                
        # Compute rollout statistics
        rollout_info = {
            "episodes_completed": episodes_completed,
            "mean_episode_reward": np.mean(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "experiences_collected": len(experiences)
        }
        
        return experiences, rollout_info
    
    def evaluate(
        self, 
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            n_episodes: Number of episodes to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        lengths = []
        
        for _ in range(n_episodes):
            # Reset
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Select action deterministically
                action = self.agent.select_action(obs, evaluate=True)
                
                # Take a step
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Update counters
                episode_reward += reward
                episode_length += 1
                
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_episode_length": np.mean(lengths)
        }