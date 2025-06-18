

import os
import json
import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime

class RewardLogger:
    """
    Logger for reward values and components.
    Records reward values over time and can save them to files.
    """
    
    def __init__(self, 
                 log_dir: str = "./reward_logs",
                 log_frequency: int = 100,
                 verbose: bool = False):
        """
        Initialize the reward logger.
        
        Args:
            log_dir: Directory where log files will be stored
            log_frequency: How often to write to disk (every N calls)
            verbose: Whether to print logging info to console
        """
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        self.verbose = verbose
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging arrays
        self.total_rewards = []
        self.component_rewards = []
        self.timestamps = []
        self.steps = []
        self.call_count = 0
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RewardLogger")
    
    def log_reward(self, 
                   total_reward: float, 
                   component_rewards: Dict[str, float],
                   step: Optional[int] = None,
                   context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a reward value.
        
        Args:
            total_reward: The total reward value
            component_rewards: Dictionary of component-wise rewards
            step: Current step number (optional)
            context: Additional context to log (optional)
        """
        timestamp = datetime.now().isoformat()
        
        self.total_rewards.append(total_reward)
        self.component_rewards.append(component_rewards)
        self.timestamps.append(timestamp)
        self.steps.append(step if step is not None else self.call_count)
        
        if self.verbose:
            self.logger.info(f"Reward: {total_reward:.4f} (Step: {self.steps[-1]})")
            
            for component, value in component_rewards.items():
                self.logger.info(f"  - {component}: {value:.4f}")
        
        self.call_count += 1
        
        # Write to disk if needed
        if self.call_count % self.log_frequency == 0:
            self.save_logs()
    
    def save_logs(self) -> None:
        """
        Save the logs to disk.
        """
        if not self.total_rewards:
            return
            
        log_data = {
            "total_rewards": self.total_rewards,
            "component_rewards": self.component_rewards,
            "timestamps": self.timestamps,
            "steps": self.steps
        }
        
        log_file = os.path.join(self.log_dir, "reward_log.json")
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f)
            
        if self.verbose:
            self.logger.info(f"Logs saved to {log_file}")
    
    def reset(self) -> None:
        """
        Reset the logger, clearing all stored values.
        """
        self.total_rewards = []
        self.component_rewards = []
        self.timestamps = []
        self.steps = []
        self.call_count = 0
        
        if self.verbose:
            self.logger.info("Logger reset")
