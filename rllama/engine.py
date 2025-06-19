# rllama/engine.py

import os
import yaml
from typing import Dict, Any
import logging
from datetime import datetime

from .rewards.composer import RewardComposer
from .rewards.shaper import RewardShaper
from .rewards.registry import REWARD_REGISTRY
from .logger import RewardLogger

class RewardEngine:
    """
    Main engine for computing composite rewards based on a configuration file.
    
    This class is responsible for:
    1. Loading and parsing reward configurations
    2. Instantiating reward components
    3. Computing rewards for given contexts
    4. Logging reward values
    """
    
    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize the reward engine.
        
        Args:
            config_path: Path to the YAML configuration file
            verbose: Whether to output verbose logging info
        """
        self.config_path = config_path
        self.verbose = verbose
        self.current_step = 0
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RewardEngine")
        
        # Parse configuration and set up components
        self._setup_components()
        
        # Create reward logger
        log_dir = self.config.get("logging", {}).get("log_dir", "./reward_logs")
        log_frequency = self.config.get("logging", {}).get("log_frequency", 100)
        self.reward_logger = RewardLogger(
            log_dir=log_dir,
            log_frequency=log_frequency,
            verbose=verbose
        )
        
        if self.verbose:
            self.logger.info(f"RewardEngine initialized with config from {config_path}")
        print("✅ RewardEngine initialized successfully.")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The parsed configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
    
    def _setup_components(self) -> None:
        """
        Set up reward components from the configuration.
        """
        # Get reward component configurations
        component_configs = self.config.get("reward_components", [])
        
        # Initialize reward components
        self.reward_components = []
        
        for comp_config in component_configs:
            # Extract component name and parameters
            name = comp_config.get("name")
            params = comp_config.get("params", {})
            
            if not name:
                self.logger.warning("Skipping component with missing name")
                continue
                
            # Look up the component class in the registry
            component_class = REWARD_REGISTRY.get(name)
            
            if component_class is None:
                self.logger.warning(f"Unknown reward component: {name}")
                continue
                
            # Instantiate the component with parameters
            try:
                component = component_class(**params)
                self.reward_components.append(component)
                
                if self.verbose:
                    self.logger.info(f"Initialized reward component: {name}")
            except Exception as e:
                self.logger.error(f"Error initializing component {name}: {e}")
        
        # Initialize composer and shaper
        self.composer = RewardComposer(self.reward_components)
        self.shaper = RewardShaper(self.config.get("shaping_config", {}))
        
    def compute(self, context: Dict[str, Any]) -> float:
        """
        Compute the reward for a context, without logging.
        
        Args:
            context: The context for reward computation
            
        Returns:
            The computed reward value
        """
        # Set current step
        if "step" not in context:
            context["step"] = self.current_step
            self.current_step += 1
        
        # Compute component rewards
        component_rewards = self.composer.calculate(context)
        
        # Get step from context if available
        step = context.get("step", 0)
        
        # Apply shaping to get final reward
        reward = self.shaper.shape(component_rewards, step)
        
        return reward
    
    def compute_and_log(self, context: Dict[str, Any]) -> float:
        """
        Compute the reward for a context and log it.
        
        Args:
            context: The context for reward computation
            
        Returns:
            The computed reward value
        """
        # Set current step if not provided
        if "step" not in context:
            context["step"] = self.current_step
            self.current_step += 1
            
        # Compute component rewards
        component_rewards = self.composer.calculate(context)
        
        # Get step from context
        step = context.get("step", self.current_step)
        
        # Apply shaping to get final reward
        reward = self.shaper.shape(component_rewards, step)
        
        # Log the reward
        self.reward_logger.log_reward(
            total_reward=reward,
            component_rewards=component_rewards,
            step=step,
            context=context
        )
        
        return reward
