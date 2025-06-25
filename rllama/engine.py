#!/usr/bin/env python3
"""
Main reward engine for RLlama framework.
Provides the primary interface for reward computation with advanced features.
"""

import os
import yaml
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
import logging
from datetime import datetime
from pathlib import Path
import warnings

from .rewards.composer import RewardComposer
from .rewards.shaper import RewardShaper
from .rewards.registry import reward_registry, REWARD_REGISTRY
from .logger import RewardLogger

class RewardEngine:
    """
    Main engine for computing composite rewards based on a configuration file.
    
    This class provides:
    - Configuration-based reward component instantiation
    - Sophisticated reward composition and shaping
    - Performance monitoring and logging
    - Dynamic component management
    - Validation and error handling
    """
    
    def __init__(self, 
                 config_path: str, 
                 verbose: bool = False,
                 validate_config: bool = True,
                 auto_save_logs: bool = True):
        """
        Initialize the reward engine.
        
        Args:
            config_path: Path to the YAML configuration file
            verbose: Whether to output verbose logging info
            validate_config: Whether to validate configuration on startup
            auto_save_logs: Whether to automatically save logs periodically
        """
        self.config_path = config_path
        self.verbose = verbose
        self.validate_config = validate_config
        self.auto_save_logs = auto_save_logs
        self.current_step = 0
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("RewardEngine")
        
        # Load and validate configuration
        self.config = self._load_config(config_path)
        
        if validate_config:
            self._validate_configuration()
        
        # Parse configuration and set up components
        self._setup_components()
        
        # Create reward logger
        self._setup_logging()
        
        # Performance tracking
        self.computation_times = []
        self.total_computations = 0
        self.error_count = 0
        self.last_error = None
        
        # Component management
        self.component_status = {}
        
        if self.verbose:
            self.logger.info(f"RewardEngine initialized with config from {config_path}")
            self.logger.info(f"Loaded {len(self.reward_components)} reward components")
            
        print("✅ RewardEngine initialized successfully.")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            The parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                config = {}
                
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in config file {config_path}: {e}")
    
    def _validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        validation_errors = []
        
        # Check required sections
        if "reward_components" not in self.config:
            validation_errors.append("Missing 'reward_components' section")
        elif not isinstance(self.config["reward_components"], list):
            validation_errors.append("'reward_components' must be a list")
            
        # Validate component configurations
        if "reward_components" in self.config:
            for i, comp_config in enumerate(self.config["reward_components"]):
                if not isinstance(comp_config, dict):
                    validation_errors.append(f"Component {i} must be a dictionary")
                    continue
                    
                if "name" not in comp_config:
                    validation_errors.append(f"Component {i} missing 'name' field")
                    continue
                    
                # Check if component exists in registry
                component_name = comp_config["name"]
                if reward_registry.get(component_name) is None:
                    validation_errors.append(f"Unknown component: {component_name}")
                    
                # Validate component-specific configuration
                validation_result = reward_registry.validate_component(component_name)
                if not validation_result["valid"]:
                    for error in validation_result["errors"]:
                        validation_errors.append(f"Component {component_name}: {error}")
        
        # Validate shaping configuration
        if "shaping_config" in self.config:
            shaping_config = self.config["shaping_config"]
            if not isinstance(shaping_config, dict):
                validation_errors.append("'shaping_config' must be a dictionary")
        
        # Report validation errors
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in validation_errors)
            raise ValueError(error_msg)
            
        if self.verbose:
            self.logger.info("Configuration validation passed")
    
    def _setup_components(self) -> None:
        """Set up reward components from the configuration."""
        # Get reward component configurations
        component_configs = self.config.get("reward_components", [])
        
        # Initialize reward components
        self.reward_components = []
        self.component_status = {}
        
        for comp_config in component_configs:
            # Extract component name and parameters
            name = comp_config.get("name")
            params = comp_config.get("params", {})
            enabled = comp_config.get("enabled", True)
            
            if not name:
                self.logger.warning("Skipping component with missing name")
                continue
                
            # Create component instance
            try:
                component = reward_registry.create(name, **params)
                
                if component is None:
                    self.logger.error(f"Failed to create component: {name}")
                    self.component_status[name] = "failed"
                    continue
                    
                # Set component properties
                if hasattr(component, 'enabled'):
                    component.enabled = enabled
                if hasattr(component, 'name'):
                    component.name = name
                    
                self.reward_components.append(component)
                self.component_status[name] = "active" if enabled else "disabled"
                
                if self.verbose:
                    self.logger.info(f"Initialized reward component: {name}")
                    
            except Exception as e:
                self.logger.error(f"Error initializing component {name}: {e}")
                self.component_status[name] = "error"
        
        # Initialize composer and shaper
        self.composer = RewardComposer(self.reward_components)
        self.shaper = RewardShaper(self.config.get("shaping_config", {}))
        
        if self.verbose:
            active_components = sum(1 for status in self.component_status.values() if status == "active")
            self.logger.info(f"Composer initialized with {active_components} active components")
    
    def _setup_logging(self) -> None:
        """Set up reward logging."""
        logging_config = self.config.get("logging", {})
        
        log_dir = logging_config.get("log_dir", "./reward_logs")
        log_frequency = logging_config.get("log_frequency", 100)
        log_format = logging_config.get("format", "json")
        
        self.reward_logger = RewardLogger(
            log_dir=log_dir,
            log_frequency=log_frequency,
            verbose=self.verbose,
            format=log_format,
            auto_save=self.auto_save_logs
        )
        
        if self.verbose:
            self.logger.info(f"Logging configured: {log_dir} (frequency: {log_frequency})")
    
    def compute(self, context: Dict[str, Any]) -> float:
        """
        Compute the reward for a context without logging.
        
        Args:
            context: The context for reward computation
            
        Returns:
            The computed reward value
        """
        start_time = time.time()
        
        try:
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
            
            # Track performance
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)
            self.total_computations += 1
            
            # Keep computation times bounded
            if len(self.computation_times) > 1000:
                self.computation_times = self.computation_times[-1000:]
            
            return reward
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            
            self.logger.error(f"Error computing reward: {e}")
            
            if self.verbose:
                self.logger.debug(f"Context that caused error: {context}")
            
            # Return 0.0 on error to avoid breaking the pipeline
            return 0.0
    
    def compute_and_log(self, context: Dict[str, Any]) -> float:
        """
        Compute the reward for a context and log it.
        
        Args:
            context: The context for reward computation
            
        Returns:
            The computed reward value
        """
        start_time = time.time()
        
        try:
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
                context=context,
                computation_time=time.time() - start_time
            )
            
            # Track performance
            computation_time = time.time() - start_time
            self.computation_times.append(computation_time)
            self.total_computations += 1
            
            # Keep computation times bounded
            if len(self.computation_times) > 1000:
                self.computation_times = self.computation_times[-1000:]
            
            return reward
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            
            self.logger.error(f"Error computing reward: {e}")
            
            if self.verbose:
                self.logger.debug(f"Context that caused error: {context}")
            
            # Return 0.0 on error
            return 0.0
    
    def add_component(self, 
                     name: str, 
                     component_class: str, 
                     params: Dict[str, Any] = None,
                     weight: float = 1.0,
                     enabled: bool = True) -> bool:
        """
        Dynamically add a reward component.
        
        Args:
            name: Name for the component instance
            component_class: Class name of the component
            params: Parameters for component initialization
            weight: Weight for the component in shaping
            enabled: Whether the component is enabled
            
        Returns:
            True if component was added successfully, False otherwise
        """
        try:
            # Create component instance
            component = reward_registry.create(component_class, **(params or {}))
            
            if component is None:
                self.logger.error(f"Failed to create component: {component_class}")
                return False
                
            # Set component properties
            if hasattr(component, 'enabled'):
                component.enabled = enabled
            if hasattr(component, 'name'):
                component.name = name
                
            # Add to components list
            self.reward_components.append(component)
            self.component_status[name] = "active" if enabled else "disabled"
            
            # Update composer
            self.composer = RewardComposer(self.reward_components)
            
            # Add to shaper configuration
            self.shaper.add_component(name, weight)
            
            if self.verbose:
                self.logger.info(f"Added component: {name} ({component_class})")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding component {name}: {e}")
            return False
    
    def remove_component(self, name: str) -> bool:
        """
        Remove a reward component by name.
        
        Args:
            name: Name of the component to remove
            
        Returns:
            True if component was removed, False if not found
        """
        # Find and remove component
        for i, component in enumerate(self.reward_components):
            component_name = getattr(component, 'name', component.__class__.__name__)
            if component_name == name:
                self.reward_components.pop(i)
                
                # Update status
                if name in self.component_status:
                    del self.component_status[name]
                
                # Update composer
                self.composer = RewardComposer(self.reward_components)
                
                # Remove from shaper
                self.shaper.remove_component(name)
                
                if self.verbose:
                    self.logger.info(f"Removed component: {name}")
                    
                return True
                
        self.logger.warning(f"Component not found for removal: {name}")
        return False
    
    def enable_component(self, name: str) -> bool:
        """Enable a component by name."""
        for component in self.reward_components:
            component_name = getattr(component, 'name', component.__class__.__name__)
            if component_name == name:
                if hasattr(component, 'enable'):
                    component.enable()
                elif hasattr(component, 'enabled'):
                    component.enabled = True
                    
                self.component_status[name] = "active"
                
                if self.verbose:
                    self.logger.info(f"Enabled component: {name}")
                    
                return True
                
        return False
    
    def disable_component(self, name: str) -> bool:
        """Disable a component by name."""
        for component in self.reward_components:
            component_name = getattr(component, 'name', component.__class__.__name__)
            if component_name == name:
                if hasattr(component, 'disable'):
                    component.disable()
                elif hasattr(component, 'enabled'):
                    component.enabled = False
                    
                self.component_status[name] = "disabled"
                
                if self.verbose:
                    self.logger.info(f"Disabled component: {name}")
                    
                return True
                
        return False
    
    def get_component_status(self) -> Dict[str, str]:
        """Get the status of all components."""
        return self.component_status.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the engine."""
        if not self.computation_times:
            return {
                "total_computations": 0,
                "avg_computation_time": 0.0,
                "error_count": self.error_count,
                "error_rate": 0.0
            }
            
        return {
            "total_computations": self.total_computations,
            "avg_computation_time": sum(self.computation_times) / len(self.computation_times),
            "min_computation_time": min(self.computation_times),
            "max_computation_time": max(self.computation_times),
            "recent_avg_time": sum(self.computation_times[-100:]) / min(100, len(self.computation_times)),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.total_computations),
            "last_error": self.last_error
        }
    
    def get_component_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed metrics for each component."""
        metrics = {}
        
        for component in self.reward_components:
            component_name = getattr(component, 'name', component.__class__.__name__)
            
            if hasattr(component, 'get_metrics'):
                metrics[component_name] = component.get_metrics()
            else:
                metrics[component_name] = {
                    "status": self.component_status.get(component_name, "unknown"),
                    "class": component.__class__.__name__
                }
                
        return metrics
    
    def get_shaping_weights(self, step: int = None) -> Dict[str, float]:
        """Get current shaping weights for all components."""
        if step is None:
            step = self.current_step
            
        return self.shaper.get_current_weights(step)
    
    def update_shaping_weights(self, weights: Dict[str, float]) -> None:
        """Update shaping weights for components."""
        for name, weight in weights.items():
            self.shaper.add_component(name, weight)
            
        if self.verbose:
            self.logger.info(f"Updated shaping weights: {weights}")
    
    def save_logs(self, path: Optional[str] = None) -> str:
        """
        Save current logs to file.
        
        Args:
            path: Optional path to save logs to
            
        Returns:
            Path where logs were saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"reward_logs_{timestamp}.json"
            
        return self.reward_logger.save_logs(path)
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.computation_times = []
        self.total_computations = 0
        self.error_count = 0
        self.last_error = None
        
        # Reset component metrics
        for component in self.reward_components:
            if hasattr(component, 'reset_metrics'):
                component.reset_metrics()
                
        # Reset logger
        self.reward_logger.reset()
        
        if self.verbose:
            self.logger.info("Reset all metrics")
    
    def export_config(self, path: str) -> None:
        """
        Export current configuration to file.
        
        Args:
            path: Path to save configuration
        """
        # Get current weights
        current_weights = self.get_shaping_weights()
        
        # Update config with current weights
        export_config = self.config.copy()
        if "shaping_config" not in export_config:
            export_config["shaping_config"] = {}
            
        export_config["shaping_config"].update(current_weights)
        
        # Add metadata
        export_config["metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "total_computations": self.total_computations,
            "component_status": self.component_status
        }
        
        # Save to file
        with open(path, 'w') as f:
            yaml.dump(export_config, f, default_flow_style=False)
            
        if self.verbose:
            self.logger.info(f"Configuration exported to: {path}")
    
    def __str__(self) -> str:
        """String representation of the engine."""
        active_components = sum(1 for status in self.component_status.values() if status == "active")
        return f"RewardEngine({active_components} active components, {self.total_computations} computations)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"RewardEngine(config='{self.config_path}', "
                f"components={len(self.reward_components)}, "
                f"computations={self.total_computations})")
