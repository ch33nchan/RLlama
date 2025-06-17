# rllama/optimization/bayesian_optimizer.py

import optuna
import yaml
from typing import Callable, Dict, Any, List

class BayesianRewardOptimizer:
    """
    Uses Optuna to perform Bayesian optimization on reward component weights.

    This class automates the process of finding the optimal weights for the
    components defined in the `shaping_config` of a RLlama YAML file.
    """
    def __init__(self, config_path: str, objective_function: Callable[[Dict], float], direction: str = "maximize"):
        """
        Initializes the BayesianRewardOptimizer.

        Args:
            config_path (str): Path to the RLlama YAML configuration file.
            objective_function (Callable[[Dict], float]): A function that takes a
                dynamically generated `shaping_config` dict, runs a training
                process, and returns a metric to be optimized (e.g., mean
                episode reward, success rate).
            direction (str): The direction of optimization. Can be "maximize" or
                "minimize".
        """
        if not hasattr(objective_function, '__call__'):
            raise TypeError("objective_function must be a callable function.")
        
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        self.objective_function = objective_function
        self.study = optuna.create_study(direction=direction)

        # Identify which components have tunable weights
        self.tunable_components = self._get_tunable_components()
        if not self.tunable_components:
            raise ValueError("No tunable components found in the shaping_config. "
                             "Mark components for tuning by adding a 'tune: true' key.")

    def _get_tunable_components(self) -> List[str]:
        """Parses the config to find components marked for tuning."""
        tunable = []
        shaping_config = self.base_config.get('shaping_config', {})
        for name, params in shaping_config.items():
            if params.get('tune') is True:
                tunable.append(name)
        return tunable

    def _objective_wrapper(self, trial: optuna.trial.Trial) -> float:
        """
        An objective function wrapper for Optuna that generates trial
        configurations and calls the user-provided objective function.
        """
        # Start with the base shaping config
        trial_shaping_config = self.base_config.get('shaping_config', {}).copy()

        # Suggest new weights for each tunable component
        for name in self.tunable_components:
            # Define search space for the weight
            low = trial_shaping_config[name].get('tune_min', 0.0)
            high = trial_shaping_config[name].get('tune_max', 1.0)
            
            # Update the weight for the current trial
            trial_shaping_config[name]['weight'] = trial.suggest_float(f"{name}_weight", low, high)
        
        # The user's function runs the actual training with this trial config
        # and returns a performance score.
        performance_metric = self.objective_function(trial_shaping_config)
        
        return performance_metric

    def optimize(self, n_trials: int = 100, show_progress_bar: bool = True):
        """
        Run the Bayesian optimization process.

        Args:
            n_trials (int): The number of optimization trials to run.
            show_progress_bar (bool): Whether to display a progress bar.
        
        Returns:
            Dict[str, Any]: The best hyperparameters (weights) found.
        """
        print(f"🚀 Starting Bayesian optimization for {n_trials} trials...")
        print(f"Optimizing weights for components: {self.tunable_components}")

        self.study.optimize(self._objective_wrapper, n_trials=n_trials, show_progress_bar=show_progress_bar)

        print("\n🎉 Optimization finished!")
        print(f"  Best trial value: {self.study.best_value:.4f}")
        print("  Best weights found:")
        for name, value in self.study.best_params.items():
            print(f"    - {name}: {value:.4f}")
            
        return self.study.best_params