import optuna
import yaml
from typing import Dict, Any, Callable, List, Tuple, Optional
import logging
import copy

# Configure logging for Optuna (optional, but helpful)
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
optuna.logging.get_logger("optuna").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

class BayesianRewardOptimizer:
    """
    Uses Optuna to perform Bayesian optimization on reward shaping hyperparameters.

    This class helps tune the 'reward_shaping' section of a reward configuration
    by optimizing parameters like initial_weight, decay_rate, decay_steps, etc.
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        search_space: Dict[str, Dict[str, Any]],
        objective_fn: Callable[[Dict[str, Any], optuna.Trial], float],
        n_trials: int = 50,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        direction: str = "maximize",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ):
        """
        Initializes the BayesianRewardOptimizer.

        Args:
            base_config: The base reward configuration dictionary (loaded from YAML or created manually).
                         This should contain at least the 'reward_components' and potentially
                         a base 'reward_shaping' section. The 'reward_shaping' section
                         will be modified during optimization based on the search_space.
            search_space: Defines the hyperparameters to tune for each reward component.
                          Format:
                          {
                              "reward_component_name": {
                                  "parameter_name": { # e.g., "initial_weight"
                                      "type": "float", # or "int", "categorical"
                                      "low": 0.1,     # for float/int
                                      "high": 10.0,   # for float/int
                                      "step": 0.1,    # optional, for float/int
                                      "log": False,   # optional, for float/int
                                      "choices": [0.1, 1.0, 10.0] # for categorical
                                  },
                                  # ... other parameters for this component ...
                              },
                              # ... other components ...
                          }
            objective_fn: A callable that takes the generated reward config (dict) and
                          an optuna.Trial object, runs the RL training/evaluation,
                          and returns a performance metric (float) to be optimized.
                          Example signature: `def my_objective(config: Dict, trial: optuna.Trial) -> float:`
            n_trials: The number of optimization trials to run.
            study_name: Name for the Optuna study. Useful for resuming studies.
            storage: Optuna storage URL (e.g., "sqlite:///db.sqlite3"). If None, uses in-memory storage.
            direction: "maximize" or "minimize" the objective function's return value.
            sampler: Optuna sampler instance (e.g., TPESampler). Defaults to Optuna's default.
            pruner: Optuna pruner instance (e.g., MedianPruner). Defaults to Optuna's default.
        """
        if "reward_shaping" not in base_config:
            base_config["reward_shaping"] = {}
        self.base_config = base_config
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.sampler = sampler
        self.pruner = pruner

        self._validate_search_space()

    def _validate_search_space(self):
        """Basic validation of the search space structure."""
        if not isinstance(self.search_space, dict):
            raise ValueError("search_space must be a dictionary.")
        for comp_name, params in self.search_space.items():
            if not isinstance(params, dict):
                raise ValueError(f"Search space for component '{comp_name}' must be a dictionary.")
            for param_name, definition in params.items():
                if not isinstance(definition, dict) or "type" not in definition:
                    raise ValueError(f"Definition for '{param_name}' in component '{comp_name}' must be a dict with a 'type'.")
                param_type = definition["type"]
                if param_type == "float" or param_type == "int":
                    if "low" not in definition or "high" not in definition:
                        raise ValueError(f"Float/Int parameter '{param_name}' in '{comp_name}' needs 'low' and 'high'.")
                elif param_type == "categorical":
                    if "choices" not in definition or not isinstance(definition["choices"], list):
                        raise ValueError(f"Categorical parameter '{param_name}' in '{comp_name}' needs 'choices' list.")
                else:
                    raise ValueError(f"Unsupported parameter type '{param_type}' for '{param_name}' in '{comp_name}'.")

    def _create_objective_wrapper(self) -> Callable[[optuna.Trial], float]:
        """Creates the objective function wrapper for Optuna."""

        # Define 'objective' as a nested function that captures 'self' from the outer scope.
        # It should only accept 'trial' as its argument from Optuna.
        def objective(trial: optuna.Trial) -> float:
            """Wrapper objective function passed to Optuna study."""
            # Create a deep copy to avoid modifying the base config between trials
            # Access 'self' from the enclosing scope (captured)
            current_config = copy.deepcopy(self.base_config)
            current_shaping = current_config.setdefault("reward_shaping", {})

            # Sample hyperparameters for this trial
            # Access 'self' from the enclosing scope (captured)
            for comp_name, params in self.search_space.items():
                if comp_name not in current_shaping:
                    # Initialize component shaping if not present in base config
                    current_shaping[comp_name] = {}

                for param_name, definition in params.items():
                    suggest_type = definition["type"]
                    optuna_param_name = f"{comp_name}_{param_name}" # Unique name for Optuna

                    if suggest_type == "float":
                        value = trial.suggest_float(
                            optuna_param_name,
                            definition["low"],
                            definition["high"],
                            step=definition.get("step"),
                            log=definition.get("log", False),
                        )
                    elif suggest_type == "int":
                         value = trial.suggest_int(
                            optuna_param_name,
                            definition["low"],
                            definition["high"],
                            step=definition.get("step", 1),
                            log=definition.get("log", False),
                        )
                    elif suggest_type == "categorical":
                        value = trial.suggest_categorical(optuna_param_name, definition["choices"])
                    else:
                        # Should be caught by validation, but defensive check
                        raise ValueError(f"Unsupported type {suggest_type}")

                    # Update the config for this trial
                    current_shaping[comp_name][param_name] = value

            # --- Add default shaping parameters if missing ---
            # Ensure essential parameters like 'decay_schedule' have defaults if not tuned
            for comp_name, shaping_params in current_shaping.items():
                if 'initial_weight' not in shaping_params:
                    logger.warning(f"Component '{comp_name}' missing 'initial_weight' in shaping config after sampling. Assuming 1.0.")
                    shaping_params['initial_weight'] = 1.0 # Default if not tuned or in base
                if 'decay_schedule' not in shaping_params:
                    # Default to 'none' if not specified in base or search space
                    shaping_params.setdefault('decay_schedule', 'none')


            # Access 'self' from the enclosing scope (captured)
            logger.info(f"Trial {trial.number}: Running with config:\n{yaml.dump({'reward_shaping': current_shaping})}")

            # Call the user-provided objective function
            try:
                # Access 'self' from the enclosing scope (captured)
                # Pass ONLY the trial object, as the user's objective_fn (the lambda) expects only that.
                score = self.objective_fn(trial) # Pass only trial
                return score
            except Exception as e:
                # Access 'self' from the enclosing scope (captured)
                # Use logger directly as it's defined at the module level or captured self.logger if defined
                logger.error(f"Trial {trial.number}: Failed with exception: {e}", exc_info=True) # Log full traceback
                raise e # Re-raising helps debug

        # Return the correctly defined nested function
        return objective

    def optimize(self) -> optuna.Study:
        """
        Runs the Optuna optimization study.

        Returns:
            The completed Optuna study object.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True, # Allows resuming studies
        )

        objective_wrapper = self._create_objective_wrapper()
        study.optimize(objective_wrapper, n_trials=self.n_trials)

        logger.info(f"Optimization finished. Number of finished trials: {len(study.trials)}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"  Value: {study.best_trial.value}")
        logger.info("  Params: ")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        return study

    def get_best_config(self, study: Optional[optuna.Study] = None) -> Dict[str, Any]:
        """
        Constructs the best reward configuration found during the study.

        Args:
            study: The Optuna study object. If None, attempts to load the study
                   using the instance's study_name and storage.

        Returns:
            The full reward configuration dictionary corresponding to the best trial.
        """
        if study is None:
            if not self.study_name or not self.storage:
                 raise ValueError("Cannot load study without study_name and storage, or provide the study object.")
            study = optuna.load_study(study_name=self.study_name, storage=self.storage)

        best_params = study.best_trial.params
        best_config = copy.deepcopy(self.base_config)
        best_shaping = best_config.setdefault("reward_shaping", {})

        for optuna_param_name, value in best_params.items():
            # Parse "component_param" back into component and param
            parts = optuna_param_name.split('_', 1)
            if len(parts) == 2:
                comp_name, param_name = parts
                if comp_name in self.search_space: # Ensure it's a param we tuned
                     if comp_name not in best_shaping:
                         best_shaping[comp_name] = {}
                     best_shaping[comp_name][param_name] = value
            else:
                logger.warning(f"Could not parse Optuna parameter name '{optuna_param_name}' back into component/parameter.")

        # Add default shaping parameters if missing (consistency check)
        for comp_name, shaping_params in best_shaping.items():
            if 'initial_weight' not in shaping_params:
                shaping_params['initial_weight'] = 1.0 # Default if not tuned or in base
            if 'decay_schedule' not in shaping_params:
                shaping_params.setdefault('decay_schedule', 'none')


        return best_config