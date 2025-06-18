# rllama/utils/config_loader.py

import yaml
from typing import Tuple
from rllama.rewards.composer import RewardComposer
from rllama.rewards.shaper import RewardShaper
from rllama.rewards.registry import REWARD_REGISTRY

def load_reward_system(config_path: str) -> Tuple[RewardComposer, RewardShaper]:
    """
    Loads reward components and configurations from a YAML file to build
    the complete reward processing system.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        A tuple containing an instance of RewardComposer and RewardShaper.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Load Reward Components ---
    reward_components_config = config.get('reward_components', [])
    if not reward_components_config:
        raise ValueError("YAML config must contain a 'reward_components' list.")

    reward_components = []
    for comp_config in reward_components_config:
        name = comp_config.get('name')
        if not name:
            raise ValueError("Each item in 'reward_components' must have a 'name'.")
        
        params = comp_config.get('params', {})
        
        if name in REWARD_REGISTRY:
            reward_class = REWARD_REGISTRY[name]
            # Instantiate the component with its parameters
            reward_components.append(reward_class(**params))
        else:
            raise ValueError(f"Reward component '{name}' not found in registry.")

    # --- Load Shaping Config ---
    shaping_config = config.get('shaping_config', {})
    if not shaping_config:
        raise ValueError("YAML config must contain a 'shaping_config' block.")

    composer = RewardComposer(reward_components)
    shaper = RewardShaper(shaping_config)

    print("✅ Reward system loaded successfully.")
    print(f"   Components: {[c.__class__.__name__ for c in composer.reward_components]}")
    print(f"   Shaping Config: {shaper.config}")
    
    return composer, shaper