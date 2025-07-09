import json
import os
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file format is not supported
    """
    # Get file extension
    _, ext = os.path.splitext(path)
    
    # Load based on file format
    if ext.lower() in [".yml", ".yaml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif ext.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
        
    Raises:
        ValueError: If file format is not supported
    """
    # Get file extension
    _, ext = os.path.splitext(path)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save based on file format
    if ext.lower() in [".yml", ".yaml"]:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    elif ext.lower() == ".json":
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported configuration file format: {ext}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add key
            result[key] = value
            
    return result


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "environment": {
            "name": "CartPole-v1",
            "normalize_obs": False,
            "normalize_reward": False,
            "time_limit": None,
        },
        "agent": {
            "name": "PPO",
            "gamma": 0.99,
            "learning_rate": 3e-4,
        },
        "training": {
            "total_steps": 100000,
            "batch_size": 64,
            "eval_freq": 10000,
            "eval_episodes": 10,
            "save_freq": 20000,
        },
        "logging": {
            "log_dir": "./logs",
            "print_freq": 1000,
            "use_wandb": False,
            "wandb_project": None,
            "wandb_entity": None,
        },
    }