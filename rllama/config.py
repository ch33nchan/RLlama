import yaml
from typing import Tuple, Dict, Any, List

# Assuming these imports will be correct based on your project structure
from .rewards.base import BaseReward
from .rewards.composition import RewardComposer
from .rewards.shaping import RewardShaper, RewardConfig
from .rewards.registry import reward_registry # Import the registry

# Define expected top-level keys in the YAML config
EXPECTED_KEYS = {"composer_settings", "reward_components", "reward_shaping"}

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

    # --- 1. Instantiate Reward Components ---
    components: List[BaseReward] = []
    component_configs = config.get("reward_components", {})
    if not isinstance(component_configs, dict):
        raise ValueError("'reward_components' section must be a dictionary (map).")

    for name, params in component_configs.items():
        if not isinstance(params, dict):
             raise ValueError(f"Parameters for component '{name}' must be a dictionary.")

        component_class_name = params.pop("class", None) # Get class name, remove from params
        if not component_class_name:
            raise ValueError(f"Missing 'class' key for reward component '{name}' in {config_path}")

        # Look up the class in the registry
        # component_class = reward_registry.get(component_class_name) # Original line
        component_class = reward_registry.get_class(component_class_name) # Corrected line
        if component_class is None:
            raise KeyError(f"Reward component class '{component_class_name}' (for '{name}') not found in the registry.")

        try:
            # Instantiate with name and other params from YAML
            component_instance = component_class(name=name, **params)
            components.append(component_instance)
            print(f"Successfully instantiated component: {name} ({component_class_name})") # Debug print
        except Exception as e:
            raise ValueError(f"Error instantiating component '{name}' ({component_class_name}): {e}")

    # Inside load_config_from_yaml function
    
        # --- 2. Instantiate Reward Shaper ---
        shaping_configs_dict = config.get("reward_shaping", {})
        if not isinstance(shaping_configs_dict, dict):
            raise ValueError("'reward_shaping' section must be a dictionary (map).")
    
        reward_configs: Dict[str, RewardConfig] = {}
        for component_name, shape_params in shaping_configs_dict.items():
             if not isinstance(shape_params, dict):
                 raise ValueError(f"Shaping parameters for component '{component_name}' must be a dictionary.")
             # TODO: Validate shape_params structure (e.g., initial_weight, schedule_type, etc.)
             try:
                 # Uses the parameters from YAML to create RewardConfig instances
                 reward_configs[component_name] = RewardConfig(**shape_params)
                 print(f"Successfully created RewardConfig for: {component_name}") # Debug print
             except Exception as e:
                 raise ValueError(f"Error creating RewardConfig for '{component_name}': {e}")
    
        # Ensure all instantiated components have a shaping config (even if default)
        # TODO: Add logic to handle components mentioned in 'reward_components' but not in 'reward_shaping' (use defaults?)
        component_names_set = {c.name for c in components}
        shaping_names_set = set(reward_configs.keys())
        if component_names_set != shaping_names_set:
            # For now, require explicit shaping config for every component
            missing_shaping = component_names_set - shaping_names_set
            extra_shaping = shaping_names_set - component_names_set
            error_msg = ""
            if missing_shaping:
                error_msg += f" Components missing shaping config: {missing_shaping}."
            if extra_shaping:
                error_msg += f" Shaping config found for non-existent components: {extra_shaping}."
            if error_msg:
                 raise ValueError(f"Mismatch between defined components and shaping configurations.{error_msg}")
    
    
    # Instantiates the RewardShaper with the created RewardConfig objects
    shaper = RewardShaper(reward_configs=reward_configs)
    print("Successfully instantiated RewardShaper.") # Debug print
    
    # --- 3. Instantiate Reward Composer ---
    composer_settings = config.get("composer_settings", {})
    if not isinstance(composer_settings, dict):
        raise ValueError("'composer_settings' section must be a dictionary.")

    # TODO: Extract specific settings like 'normalization_strategy' etc.
    normalization_strategy = composer_settings.get("normalization_strategy", None) # Example

    composer = RewardComposer(
        reward_components=components,
        normalization_strategy=normalization_strategy
        # Pass other composer settings as needed
    )
    print("Successfully instantiated RewardComposer.") # Debug print

    print(f"Configuration loaded successfully from {config_path}")
    return composer, shaper # Returns both composer and shaper

# Example Usage (for testing purposes, can be removed later)
if __name__ == '__main__':
    # Create a dummy YAML file for testing
    dummy_yaml_content = """
composer_settings:
  normalization_strategy: null # Or 'min_max', 'z_score' etc.

reward_components:
  preference:
    class: PreferenceScoreReward # Assumes this is registered
    model_path: "/path/to/pref/model"
  toxicity:
    class: ToxicityPenalty # Assumes this is registered
    penalty_value: -1.0
    threshold: 0.9

reward_shaping:
  preference:
    initial_weight: 1.0
    schedule_type: constant
  toxicity:
    initial_weight: 0.5
    schedule_type: linear_decay
    decay_rate: 0.0001
    min_weight: 0.1
"""
    dummy_path = "/tmp/dummy_reward_config.yaml"
    with open(dummy_path, 'w') as f:
        f.write(dummy_yaml_content)

    # --- Mock necessary classes/registry for standalone testing ---
    # We still need mocks for the base classes and potentially others
    # if the real ones have complex dependencies not met here.

    # Import the real classes to ensure they are registered IF NOT ALREADY IMPORTED
    # (The import in rllama.rewards.__init__ should handle this if config.py imports from rllama.rewards)
    try:
        from .rewards.specific_rewards import ToxicityPenalty, PreferenceScoreReward
        print("Successfully imported real reward components for registration check.")
    except ImportError:
        print("Could not import real reward components directly, assuming registry is populated elsewhere.")
        # If imports fail here, rely on the __init__.py import mechanism

    # Mocks for classes NOT being tested directly
    class MockRewardConfig(RewardConfig):
         def __init__(self, initial_weight: float = 1.0, schedule_type: str = 'constant', **kwargs):
             self.initial_weight = initial_weight
             self.schedule_type = schedule_type
             self.params = kwargs
             print(f"Initialized MockRewardConfig with initial_weight={initial_weight}, schedule_type='{schedule_type}'") # Debug print
         def calculate_weight(self, global_step: int) -> float:
             return self.initial_weight # Dummy calculation

    class MockRewardShaper(RewardShaper):
        def __init__(self, reward_configs: Dict[str, RewardConfig]):
            self._configs = reward_configs
            self._weights = {name: cfg.initial_weight for name, cfg in reward_configs.items()}
            print(f"Initialized MockRewardShaper with configs for: {list(reward_configs.keys())}") # Debug print
        def update_weights(self, global_step: int): pass # No-op for mock
        def get_weights(self) -> Dict[str, float]: return self._weights

    class MockRewardComposer(RewardComposer):
        def __init__(self, reward_components: List[BaseReward], normalization_strategy: str = None):
            self.components = reward_components
            self.norm_strat = normalization_strategy
            print(f"Initialized MockRewardComposer with components: {[c.name for c in components]}") # Debug print
        def compute_rewards(self, state, action, next_state, info) -> Dict[str, float]:
            print(f"MockRewardComposer computing rewards with info: {info}") # Debug print
            return {c.name: c(state, action, next_state, info) for c in self.components}
        def combine_rewards(self, raw_rewards: Dict[str, float], weights: Dict[str, float]) -> float:
            print(f"MockRewardComposer combining rewards: {raw_rewards} with weights: {weights}") # Debug print
            return sum(raw_rewards.get(name, 0.0) * weights.get(name, 0.0) for name in weights)

    # Replace only the classes we AREN'T testing the instantiation of
    # RewardComposer = MockRewardComposer # Keep real RewardComposer if its init is simple
    RewardShaper = MockRewardShaper     # Keep real RewardShaper if its init is simple
    RewardConfig = MockRewardConfig   # Use MockRewardConfig as the real one might have complex logic

    # --- Ensure registry has the REAL classes (the decorator should handle this) ---
    # Remove the explicit mock registrations for the classes we implemented:
    # reward_registry.register("PreferenceScoreReward", MockBaseReward) # REMOVE/COMMENT OUT
    # reward_registry.register("ToxicityPenalty", MockBaseReward)       # REMOVE/COMMENT OUT
    print("\nRegistry content check:")
    print(reward_registry._registry) # Print registry content for debugging
    # --- End Mocking Adjustments ---

    try:
        print(f"\nAttempting to load config from: {dummy_path} using potentially real components")
        # We expect load_config_from_yaml to use the REAL registered classes now
        composer, shaper = load_config_from_yaml(dummy_path)

        print("\n--- Loaded Objects ---")
        print("Composer Type:", type(composer))
        print("Shaper Type:", type(shaper))
        if hasattr(composer, 'components'):
             print("Composer Components:", [(c.name, type(c)) for c in composer.components])
        if hasattr(shaper, '_configs'): # Accessing protected member for debug
             print("Shaper Configs:", [(name, type(cfg)) for name, cfg in shaper._configs.items()])


        print("\n--- Testing Usage ---")
        weights = shaper.get_weights() # Should use MockRewardShaper.get_weights
        print("Initial Weights:", weights)

        # Provide necessary info for the real ToxicityPenalty component
        test_info = {'toxicity_score': 0.95} # Example score above threshold
        print(f"Calling composer.compute_rewards with info: {test_info}")
        raw = composer.compute_rewards(None, "test action", None, test_info)
        print("Raw Rewards:", raw)

        final = composer.combine_rewards(raw, weights)
        print("Final Reward:", final)

        # Test case where toxicity is below threshold
        test_info_low_toxicity = {'toxicity_score': 0.5}
        print(f"\nCalling composer.compute_rewards with info: {test_info_low_toxicity}")
        raw_low_toxicity = composer.compute_rewards(None, "test action", None, test_info_low_toxicity)
        print("Raw Rewards (Low Toxicity):", raw_low_toxicity)
        final_low_toxicity = composer.combine_rewards(raw_low_toxicity, weights)
        print("Final Reward (Low Toxicity):", final_low_toxicity)


    except (ValueError, FileNotFoundError, KeyError, TypeError) as e: # Added TypeError
        print(f"\nError loading or testing configuration: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for errors