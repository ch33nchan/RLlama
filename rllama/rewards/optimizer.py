import optuna
import logging
import yaml
from typing import Dict, Any, Callable, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class BayesianRewardOptimizer:
    """
    Uses Optuna for Bayesian optimization of reward shaping parameters.
    """
    def __init__(
        self,
        objective_function: Callable[[optuna.Trial, Dict[str, Any], Dict[str, Any]], float],
        base_config: Dict[str, Any],
        search_space: Dict[str, Dict[str, Any]],
        n_trials: int = 100,
        study_name: Optional[str] = "reward_optimization",
        storage: Optional[str] = None, # e.g., "sqlite:///db.sqlite3"
        direction: str = "maximize", # or "minimize"
        sampler: Optional[optuna.samplers.BaseSampler] = None, # e.g., TPESampler
        pruner: Optional[optuna.pruners.BasePruner] = None, # e.g., MedianPruner
    ):
        """
        Initializes the Bayesian Reward Optimizer.

        Args:
            objective_function: A function that takes an Optuna trial, the base config,
                                and the search space, runs an RL experiment with sampled
                                parameters, and returns a performance metric (e.g., total reward).
                                Signature: objective(trial, base_config, search_space) -> float
            base_config: The base configuration dictionary (loaded from YAML).
            search_space: A dictionary defining the parameters to tune and their ranges/choices.
                          Example: {'component_name': {'param_name': {'type': 'float', 'low': 0.1, 'high': 1.0}}}
            n_trials: The number of optimization trials to run.
            study_name: Name for the Optuna study.
            storage: Database URL for Optuna study persistence. If None, uses in-memory storage.
            direction: Direction of optimization ('maximize' or 'minimize').
            sampler: Optuna sampler instance. Defaults to Optuna's default (TPE).
            pruner: Optuna pruner instance. Defaults to Optuna's default (no pruning).
        """
        if not callable(objective_function):
            raise TypeError("objective_function must be a callable")
        if not isinstance(base_config, dict):
            raise TypeError("base_config must be a dictionary")
        if not isinstance(search_space, dict):
            raise TypeError("search_space must be a dictionary")

        self.objective_function = objective_function
        self.base_config = base_config
        self.search_space = search_space
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.sampler = sampler
        self.pruner = pruner

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True # Load existing study if name and storage match
        )

        logger.info(f"Optuna study '{self.study_name}' created/loaded. Direction: {self.direction}")
        logger.info(f"Sampler: {self.study.sampler.__class__.__name__}, Pruner: {self.study.pruner.__class__.__name__}")
        logger.info(f"Number of finished trials in study: {len(self.study.trials)}")


    def _objective_wrapper(self, trial: optuna.Trial) -> float:
        """Wrapper to pass necessary arguments to the user-defined objective function."""
        try:
            # Pass the base config and search space along with the trial
            performance = self.objective_function(trial, self.base_config, self.search_space)
            logger.info(f"Trial {trial.number} finished with value: {performance:.4f}")
            # You could add logging of sampled parameters here if desired
            # logger.info(f"Trial {trial.number} params: {trial.params}")
            return performance
        except optuna.exceptions.TrialPruned as e:
             logger.info(f"Trial {trial.number} pruned.")
             raise e # Re-raise to let Optuna handle pruning
        except Exception as e:
            logger.error(f"Error during Trial {trial.number}: {e}", exc_info=True)
            # Return a very bad value or re-raise depending on desired behavior
            # Returning a bad value allows optimization to continue
            return -float('inf') if self.direction == "maximize" else float('inf')


    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Runs the Optuna optimization process.

        Returns:
            A tuple containing:
            - best_params: Dictionary of the best hyperparameters found.
            - best_value: The best objective function value achieved.
        """
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        try:
            self.study.optimize(self._objective_wrapper, n_trials=self.n_trials, timeout=None) # No timeout unless specified
        except KeyboardInterrupt:
             logger.warning("Optimization interrupted by user.")
        except Exception as e:
             logger.error(f"An unexpected error occurred during optimization: {e}", exc_info=True)

        if not self.study.trials:
             logger.warning("No trials were completed. Cannot determine best parameters.")
             return {}, -float('inf') if self.direction == "maximize" else float('inf')

        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        logger.info("Optimization finished.")
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best value: {best_value:.4f}")
        logger.info("Best parameters found:")
        logger.info(yaml.dump(best_params, default_flow_style=False))

        return best_params, best_value

    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best parameters found so far."""
        try:
            return self.study.best_params
        except ValueError:
            logger.warning("No completed trials found in the study yet.")
            return {}

    def get_best_value(self) -> float:
        """Returns the best objective value found so far."""
        try:
            return self.study.best_value
        except ValueError:
             logger.warning("No completed trials found in the study yet.")
             return -float('inf') if self.direction == "maximize" else float('inf')

    def get_study(self) -> optuna.Study:
        """Returns the underlying Optuna study object."""
        return self.study