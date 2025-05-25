# rllama/utils/config_loader.py

from typing import Tuple, Dict, Any, List, Optional
import yaml
import logging

# Assuming rllama.config and rllama.rewards.registry are accessible
# from rllama.config import load_config_from_yaml # We will implement the logic here instead
from rllama.rewards.registry import reward_registry
from rllama.rewards.composition import RewardComposer
from rllama.rewards.shaping import RewardShaper
from rllama.rewards.base import BaseReward # Import BaseReward for type hinting

logger = logging.getLogger(__name__)

# Attempt to import LLM components; they might not exist if this is a base setup
try:
    from rllama.rewards import llm_components
except ImportError:
    llm_components = None # Handle gracefully if llm_components.py doesn't exist yet

_llm_components_registered = False

def register_llm_components_if_not_present():
    """
    Registers predefined LLM-specific reward components with the reward_registry
    if they haven't been registered already.
    This function should be called before loading configurations that might
    use these components.
    """
    global _llm_components_registered
    if _llm_components_registered:
        return

    if llm_components:
        components_to_register = {
            "CoherenceReward": getattr(llm_components, "CoherenceReward", None),
            "ConcisionReward": getattr(llm_components, "ConcisionReward", None),
            "DiversityReward": getattr(llm_components, "DiversityReward", None),
            "FactualityReward": getattr(llm_components, "FactualityReward", None),
            # Add other LLM components here by their class name string
        }

        registered_any = False
        for name, component_class in components_to_register.items():
            if component_class:
                if not reward_registry.is_registered(name): # Check before registering
                    reward_registry.register(name, component_class)
                    registered_any = True
                elif reward_registry.get_class(name) != component_class:
                    # If a different class is registered with the same name, it's a conflict.
                    # This could happen if user registers a custom component with a clashing name.
                    logger.warning(f"Component name '{name}' is already registered with a different class. Skipping registration of default LLM component.")
        
        if registered_any:
            logger.info("Registered standard LLM reward components with the reward_registry.")
        _llm_components_registered = True # Mark as attempted/done even if some were skipped
    else:
        logger.info("Note: rllama.rewards.llm_components module not found. Skipping registration of default LLM components.")
        _llm_components_registered = True # Avoid re-attempting if module is missing


class RewardConfigLoader:
    """
    Handles loading of RLlama configurations from YAML files and creating
    the core RewardComposer and RewardShaper instances.
    """
    def __init__(self, config_path: str):
        """
        Initializes the RewardConfigLoader.

        Args:
            config_path (str): Path to the RLlama YAML configuration file.
        """
        self.config_path = config_path
        self._raw_config_dict = None # To store the loaded YAML dict if needed

    def _load_raw_config_if_needed(self) -> Dict[str, Any]:
        """Loads the raw YAML into a dictionary if not already loaded."""
        if self._raw_config_dict is None:
            try:
                with open(self.config_path, 'r') as f:
                    self._raw_config_dict = yaml.safe_load(f)
                if not isinstance(self._raw_config_dict, dict):
                    raise ValueError(f"YAML content in {self.config_path} must be a dictionary.")
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file {self.config_path}: {e}")
        return self._raw_config_dict

    def load_config(self) -> Dict[str, Any]:
        """
        Loads the YAML configuration file into a dictionary.
        This method primarily ensures the config is loaded and accessible.
        The actual instantiation of components happens in create_composer/shaper.

        Returns:
            Dict[str, Any]: The parsed configuration dictionary.
        """
        return self._load_raw_config_if_needed()

    def create_composer_and_shaper(self) -> Tuple[RewardComposer, RewardShaper]:
        """
        Loads the configuration from the YAML file specified during initialization
        and creates configured RewardComposer and RewardShaper instances.

        Returns:
            Tuple[RewardComposer, RewardShaper]: The configured composer and shaper.
        """
        # Ensure LLM components are registered if they exist and haven't been
        register_llm_components_if_not_present()

        config = self._load_raw_config_if_needed()

        # 1. Instantiate components
        components_config = config.get('components', [])
        component_instances_list: List[BaseReward] = [] # Changed variable name for clarity
        for comp_cfg in components_config:
            # Make a copy to avoid modifying the original config dict during pop
            comp_cfg_copy = comp_cfg.copy()
            comp_type = comp_cfg_copy.pop('type')
            comp_name = comp_cfg_copy.pop('name', comp_type) # Pop name, default to type

            # Extract the 'params' dictionary, if it exists
            component_params = comp_cfg_copy.pop('params', {})

            # Extract weight from the 'config' sub-dictionary if present
            weight = 1.0 # Default weight
            if 'config' in comp_cfg_copy and isinstance(comp_cfg_copy['config'], dict):
                weight = comp_cfg_copy['config'].get('weight', 1.0)
            comp_cfg_copy.pop('config', None) # Remove 'config' after extracting weight

            try:
                comp_class = reward_registry.get_class(comp_type)
                if comp_class is None:
                    available = []
                    if hasattr(reward_registry, '_registry'):
                        available = list(reward_registry._registry.keys())
                    elif hasattr(reward_registry, 'registry'):
                        available = list(reward_registry.registry.keys())
                    
                    raise ValueError(
                        f"Unknown reward component type: '{comp_type}'. "
                        f"Available types: {', '.join(available) if available else 'None found'}"
                    )
                
                if not hasattr(comp_class, '__init__'):
                    raise ValueError(f"Component class {comp_type} is not properly implemented")
                                    
                # Corrected line: Use component_params instead of params_section
                init_kwargs = {**component_params, **comp_cfg_copy}

                # Ensure component_instance is created correctly
                component_instance = comp_class(name=comp_name, weight=weight, **init_kwargs)
                component_instances_list.append(component_instance)
                logger.info(f"Successfully instantiated component: {comp_name} ({comp_type}) with params: {init_kwargs}")

            except ValueError as e:
                logger.error(f"Configuration error for component {comp_name}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error initializing component {comp_name}: {e}")
                raise

        # Convert list of component instances to Dict[str, BaseReward] for RewardComposer
        # The RewardComposer expects a dictionary where keys are component names.
        # The BaseReward instances already have a .name attribute.
        composer_components_dict: Dict[str, BaseReward] = { 
            comp.name: comp for comp in component_instances_list 
        }

        # Extract weights for the RewardComposer from the original component configurations
        # The RewardComposer's __init__ takes an optional 'weights' dict.
        # The 'config' section in the YAML for each component contains its weight.
        composer_weights: Dict[str, float] = {}
        for comp_cfg in components_config: # Iterate over the raw component configs from YAML
            comp_name = comp_cfg.get('name', comp_cfg.get('type')) # Get name as defined in YAML
            weight = 1.0 # Default
            if 'config' in comp_cfg and isinstance(comp_cfg['config'], dict):
                weight = comp_cfg['config'].get('weight', 1.0)
            if comp_name: # Ensure comp_name is valid
                 composer_weights[comp_name] = weight

        # 2. Get composer settings (these are currently not used by RewardComposer's __init__)
        composer_settings = config.get('composer_settings', {})
        # composer_normalization_strategy = composer_settings.get('normalization_strategy', 'none')
        # composer_norm_window = composer_settings.get('norm_window', 100)
        # composer_epsilon = composer_settings.get('epsilon', 1e-8)
        # composer_clip_range = composer_settings.get('clip_range')
        # composer_extra_kwargs = {k: v for k, v in composer_settings.items() if k not in [
        #     'normalization_strategy', 'norm_window', 'epsilon', 'clip_range'
        # ]}

        # 3. Instantiate RewardComposer
        try:
            # Pass the dictionary of components and the extracted weights
            composer = RewardComposer(
                components=composer_components_dict,
                weights=composer_weights
            )
            logger.info("Successfully instantiated RewardComposer.")
        except Exception as e:
            logger.error(f"Error instantiating RewardComposer: {e}", exc_info=True)
            raise

        # 4. Get shaper settings and instantiate RewardShaper
        shaper_settings = config.get('shaper_settings', {})
        
        # Create RewardConfig objects for each component
        from rllama.rewards.shaping import RewardConfig
        component_reward_configs = {}
        
        for comp_cfg in components_config:
            comp_name = comp_cfg.get('name', comp_cfg.get('type'))
            # Extract shaping configuration from the component config
            shaping_config = comp_cfg.get('shaping', {})
            
            # Create RewardConfig with default values if not specified
            reward_config = RewardConfig(
                name=comp_name,
                initial_weight=shaping_config.get('initial_weight', 1.0),
                decay_schedule=shaping_config.get('decay_schedule', 'none'),
                decay_rate=shaping_config.get('decay_rate', 0.0),
                decay_steps=shaping_config.get('decay_steps', 0),
                min_weight=shaping_config.get('min_weight', 0.0),
                max_weight=shaping_config.get('max_weight', float('inf')),
                start_step=shaping_config.get('start_step', 0)
            )
            component_reward_configs[comp_name] = reward_config
        
        # Instantiate Shaper with component_reward_configs as first argument
        try:
             shaper = RewardShaper(component_reward_configs, **shaper_settings)
             logger.info("Successfully instantiated RewardShaper.")
        except Exception as e:
             logger.error(f"Error instantiating RewardShaper: {e}", exc_info=True)
             raise # Re-raise the error


        return composer, shaper

    def create_composer(self, config_dict: Dict[str, Any] = None) -> RewardComposer:
        """
        Creates a RewardComposer based on the configuration.
        If config_dict is not provided, it loads from self.config_path.

        Args:
            config_dict (Dict[str, Any], optional): A pre-loaded configuration dictionary.
                If None, the configuration will be loaded from self.config_path.

        Returns:
            RewardComposer: The configured RewardComposer.
        """
        # Ensure LLM components are registered if they exist and haven't been
        register_llm_components_if_not_present()

        config = config_dict if config_dict is not None else self._load_raw_config_if_needed()

        # 1. Instantiate components (same logic as create_composer_and_shaper)
        components_config = config.get('components', [])
        components: List[BaseReward] = []
        for comp_cfg in components_config:
            comp_cfg_copy = comp_cfg.copy() 
            comp_type = comp_cfg_copy.pop('type') 
            comp_name = comp_cfg_copy.pop('name', comp_type)

            try:
                comp_class = reward_registry.get(comp_type)
                if comp_class is None:
                     raise ValueError(f"Unknown reward component type: {comp_type}")
                component_instance = comp_class(name=comp_name, **comp_cfg_copy)
                components.append(component_instance)
                logger.info(f"Successfully instantiated component: {comp_name} ({comp_type})")
            except Exception as e:
                logger.error(f"Error instantiating reward component {comp_name} ({comp_type}): {e}", exc_info=True)
                raise 

        # 2. Get composer settings
        composer_settings = config.get('composer_settings', {})
        
        # --- MODIFIED SECTION: Explicitly extract and pass arguments ---
        composer_components = components
        composer_normalization_strategy = composer_settings.get('normalization_strategy', 'none')
        composer_norm_window = composer_settings.get('norm_window', 100)
        composer_epsilon = composer_settings.get('epsilon', 1e-8)
        composer_clip_range = composer_settings.get('clip_range')

        composer_extra_kwargs = {k: v for k, v in composer_settings.items() if k not in [
            'components',
            'normalization_strategy',
            'norm_window',
            'epsilon',
            'clip_range'
        ]}

        # 3. Instantiate RewardComposer
        try:
            composer = RewardComposer(
                components=composer_components,
                normalization_strategy=composer_normalization_strategy,
                norm_window=composer_norm_window,
                epsilon=composer_epsilon,
                clip_range=composer_clip_range,
                **composer_extra_kwargs
            )
            logger.info("Successfully instantiated RewardComposer.")
            return composer
        except Exception as e:
            logger.error(f"Error instantiating RewardComposer: {e}", exc_info=True)
            raise 


    def create_shaper(self, config_dict: Dict[str, Any] = None) -> RewardShaper:
        """
        Creates a RewardShaper based on the configuration.
        If config_dict is not provided, it loads from self.config_path.

        Args:
            config_dict (Dict[str, Any], optional): A pre-loaded configuration dictionary.
                If None, the configuration will be loaded from self.config_path.

        Returns:
            RewardShaper: The configured RewardShaper.
        """
        # Ensure LLM components are registered if they exist and haven't been
        register_llm_components_if_not_present()

        config = config_dict if config_dict is not None else self._load_raw_config_if_needed()

        # Get shaper settings and instantiate RewardShaper
        shaper_settings = config.get('shaper_settings', {})
        
        # Filter out normalization_strategy from shaper settings as it's not a valid parameter for RewardShaper
        shaper_kwargs = {k: v for k, v in shaper_settings.items() 
                         if k != 'normalization_strategy'}
        
        # Instantiate Shaper with filtered kwargs
        try:
             shaper = RewardShaper(**shaper_kwargs)
             logger.info("Successfully instantiated RewardShaper.")
             return shaper
        except Exception as e:
             logger.error(f"Error instantiating RewardShaper: {e}", exc_info=True)
             raise