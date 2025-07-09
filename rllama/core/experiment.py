import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

from rllama.utils.logger import Logger


class Experiment:
    """
    Central class for managing reinforcement learning experiments.
    
    Attributes:
        name: Name of the experiment
        agent: RL agent to use
        env: Environment to train and evaluate on
        config: Configuration dictionary
        logger: Logger for tracking metrics
        checkpoint_dir: Directory to save checkpoints
        save_freq: Frequency to save checkpoints (in training steps)
        eval_freq: Frequency to run evaluations (in training steps)
        device: Device to run the experiment on
    """
    
    def __init__(
        self,
        name: str,
        agent: Any,
        env: Any,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        checkpoint_dir: str = "./checkpoints",
        save_freq: int = 10000,
        eval_freq: int = 5000,
        device: str = "auto",
    ):
        """
        Initialize an experiment.
        
        Args:
            name: Name of the experiment
            agent: RL agent to use
            env: Environment to train and evaluate on
            config: Configuration dictionary
            logger: Logger for tracking metrics
            checkpoint_dir: Directory to save checkpoints
            save_freq: Frequency to save checkpoints (in training steps)
            eval_freq: Frequency to run evaluations (in training steps)
            device: Device to run the experiment on
        """
        self.name = name
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger or Logger(name=name)
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save configuration
        self.save_config()
        
    def train(
        self, 
        total_steps: int, 
        eval_episodes: int = 10,
        progress_bar: bool = True
    ) -> Dict[str, Any]:
        """
        Train the agent for a specified number of steps.
        
        Args:
            total_steps: Total number of training steps
            eval_episodes: Number of episodes to evaluate on during evaluation
            progress_bar: Whether to display a progress bar
            
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        training_metrics = {"steps": [], "rewards": [], "losses": []}
        
        self.logger.info(f"Starting training for {total_steps} steps")
        
        if progress_bar:
            try:
                from tqdm import tqdm
                steps_iter = tqdm(range(total_steps), desc="Training")
            except ImportError:
                self.logger.warning("tqdm not found, progress bar disabled")
                steps_iter = range(total_steps)
                progress_bar = False
        else:
            steps_iter = range(total_steps)
        
        step = 0
        while step < total_steps:
            # Train for one step
            metrics = self.agent.train_step()
            step += 1
            
            # Update progress bar
            if progress_bar:
                steps_iter.update(1)
                steps_iter.set_postfix({
                    "reward": f"{metrics.get('episode_reward', 0):.2f}",
                    "loss": f"{metrics.get('loss', 0):.4f}"
                })
            
            # Log metrics
            for key, value in metrics.items():
                self.logger.record(key, value)
            
            # Store metrics
            training_metrics["steps"].append(step)
            training_metrics["rewards"].append(metrics.get("episode_reward", 0))
            training_metrics["losses"].append(metrics.get("loss", 0))
            
            # Evaluate if needed
            if step % self.eval_freq == 0:
                eval_metrics = self.evaluate(eval_episodes)
                self.logger.info(f"Step {step}/{total_steps}: Mean reward: {eval_metrics['mean_reward']:.2f}")
                
                for key, value in eval_metrics.items():
                    self.logger.record(f"eval/{key}", value)
            
            # Save checkpoint if needed
            if step % self.save_freq == 0:
                self.save_checkpoint(step)
                
        # Final evaluation
        eval_metrics = self.evaluate(eval_episodes)
        self.logger.info(f"Final evaluation: Mean reward: {eval_metrics['mean_reward']:.2f}")
        
        for key, value in eval_metrics.items():
            self.logger.record(f"eval/{key}", value)
        
        # Final checkpoint
        self.save_checkpoint(total_steps)
        
        # Log training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.record("total_training_time", total_time)
        
        return {
            "training_metrics": training_metrics,
            "final_eval_metrics": eval_metrics,
            "total_time": total_time
        }
    
    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent on the environment.
        
        Args:
            episodes: Number of episodes to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        rewards = []
        episode_lengths = []
        
        for _ in range(episodes):
            episode_reward = 0
            episode_length = 0
            obs, _ = self.env.reset()
            done = False
            
            while not done:
                action = self.agent.select_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                obs = next_obs
                
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_episode_length": np.mean(episode_lengths)
        }
    
    def save_checkpoint(self, step: int) -> str:
        """
        Save a checkpoint of the agent.
        
        Args:
            step: Current training step
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.name}_step_{step}.pt"
        )
        
        checkpoint = {
            "step": step,
            "agent_state": self.agent.state_dict(),
            "config": self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint
            
        Returns:
            Step of the loaded checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent_state"])
        
        self.logger.info(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
        
        return checkpoint["step"]
    
    def save_config(self) -> str:
        """
        Save the experiment configuration to a YAML file.
        
        Returns:
            Path to the saved configuration file
        """
        config_path = os.path.join(self.checkpoint_dir, f"{self.name}_config.yaml")
        
        # Prepare config for saving
        save_config = self.config.copy()
        save_config["experiment"] = {
            "name": self.name,
            "device": self.device,
            "save_freq": self.save_freq,
            "eval_freq": self.eval_freq
        }
        
        with open(config_path, "w") as f:
            yaml.dump(save_config, f, default_flow_style=False)
            
        return config_path
    
    @classmethod
    def from_config(
        cls, 
        config_path: str,
        checkpoint_path: Optional[str] = None
    ) -> "Experiment":
        """
        Create an experiment from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            checkpoint_path: Optional path to a checkpoint to load
            
        Returns:
            Initialized experiment
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        # Extract experiment config
        exp_config = config.pop("experiment", {})
        name = exp_config.get("name", "experiment")
        device = exp_config.get("device", "auto")
        save_freq = exp_config.get("save_freq", 10000)
        eval_freq = exp_config.get("eval_freq", 5000)
        
        # Create environment
        from rllama.environments import make_env
        env_name = config.get("environment", {}).get("name", "CartPole-v1")
        env = make_env(env_name, **config.get("environment", {}).get("params", {}))
        
        # Create agent
        from rllama.agents import make_agent
        agent_name = config.get("agent", {}).get("name", "PPO")
        agent = make_agent(
            agent_name,
            env,
            **config.get("agent", {}).get("params", {})
        )
        
        # Create experiment
        experiment = cls(
            name=name,
            agent=agent,
            env=env,
            config=config,
            device=device,
            save_freq=save_freq,
            eval_freq=eval_freq
        )
        
        # Load checkpoint if provided
        if checkpoint_path:
            experiment.load_checkpoint(checkpoint_path)
            
        return experiment