import yaml
from typing import Tuple, Dict, Any, List

# Assuming these imports will be correct based on your project structure
from .rewards.base import BaseReward
from .rewards.composition import RewardComposer
from .rewards.shaping import RewardShaper, RewardConfig
from .rewards.registry import reward_registry # Import the registry

# Define expected top-level keys in the YAML config
# Updated to reflect the structure used in rllama_config_trl.yaml
EXPECTED_KEYS = {"composer_settings", "shaper_settings", "components"}

def load_config_from_yaml(config_path: str) -> Tuple[RewardComposer, RewardShaper]:
    """
    Loads the reward configuration from a YAML file and instantiates
    the RewardComposer and RewardShaper.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A tuple containing the configured RewardComposer and RewardShaper.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed or missing required sections.
        KeyError: If a specified reward component class is not found in the registry.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_path}: {e}")

    if not isinstance(config, dict):
         raise ValueError(f"YAML content in {config_path} must be a dictionary.")

    # --- Validate Top-Level Structure ---
    missing_keys = EXPECTED_KEYS - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required sections in {config_path}: {', '.join(missing_keys)}")

    # --- 1. Instantiate Reward Components & Collect their RewardConfigs ---
    components_list_config = config.get("components", [])
    if not isinstance(components_list_config, list):
        raise ValueError("'components' section must be a list.")

    initialized_components: List[BaseReward] = []
    reward_configs_from_components: Dict[str, RewardConfig] = {}

    for comp_dict in components_list_config:
        if not isinstance(comp_dict, dict):
            raise ValueError(f"Each item in 'components' list must be a dictionary. Found: {comp_dict}")

        name = comp_dict.get("name")
        component_class_name = comp_dict.get("class")
        component_params = comp_dict.get("params", {}) # Optional
        individual_reward_config_dict = comp_dict.get("config", {}) # Shaping config for this component

        if not name:
            raise ValueError(f"Missing 'name' key for a component in {config_path}: {comp_dict}")
        if not component_class_name:
            raise ValueError(f"Missing 'class' key for reward component '{name}' in {config_path}")
        if not isinstance(component_params, dict):
            raise ValueError(f"'params' for component '{name}' must be a dictionary.")
        if not isinstance(individual_reward_config_dict, dict):
            raise ValueError(f"'config' for component '{name}' (for shaping) must be a dictionary.")

        component_class = reward_registry.get_class(component_class_name)
        if component_class is None:
            raise KeyError(f"Reward component class '{component_class_name}' (for '{name}') not found in the registry.")

        try:
            # Instantiate component with its specific params
            component_instance = component_class(name=name, **component_params)
            initialized_components.append(component_instance)
            print(f"Successfully instantiated component: {name} ({component_class_name})")
        except Exception as e:
            raise ValueError(f"Error instantiating component '{name}' ({component_class_name}): {e}")

        try:
            # Create RewardConfig for this component from its 'config' block
            # Ensure 'name' is not passed to RewardConfig constructor if it's not an arg
            reward_configs_from_components[name] = RewardConfig(**individual_reward_config_dict)
            print(f"Successfully created RewardConfig for: {name} from component's 'config' block")
        except Exception as e:
            raise ValueError(f"Error creating RewardConfig for '{name}' from its 'config' block: {e}")


    # --- 2. Instantiate Reward Shaper ---
    # Global shaper settings
    shaper_settings_dict = config.get("shaper_settings", {})
    if not isinstance(shaper_settings_dict, dict):
        raise ValueError("'shaper_settings' section must be a dictionary.")

    # The reward_configs are now primarily sourced from each component's 'config' block.
    # The shaper_settings_dict is for global shaper parameters like output normalization,
    # default weight (if any), and composition strategy.

    # Pass both the component-specific RewardConfig instances and global shaper settings
    shaper = RewardShaper(
        component_reward_configs=reward_configs_from_components, 
        **shaper_settings_dict # Pass global settings directly to RewardShaper constructor
    )
    print("Successfully instantiated RewardShaper.")

    # --- 3. Instantiate Reward Composer ---
    composer_settings = config.get("composer_settings", {})
    if not isinstance(composer_settings, dict):
        raise ValueError("'composer_settings' section must be a dictionary.")

    # Pass composer settings directly
    composer = RewardComposer(
        components=initialized_components, # MODIFIED: Changed 'reward_components' to 'components'
        **composer_settings # Pass global composer settings
    )
    print("Successfully instantiated RewardComposer.")

    print(f"Configuration loaded successfully from {config_path}")
    return composer, shaper