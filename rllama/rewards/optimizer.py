# rllama/rewards/optimizer.py

import os
import yaml
import numpy as np
from typing import Dict, Any, Callable, Optional, Union, List, Tuple
from datetime import datetime
import logging

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class OptimizerResults:
    """Container for optimization results"""
    
    def __init__(self, best_params: Dict[str, float], best_value: float):
        self.best_params = best_params
        self.best_value = best_value

class BayesianRewardOptimizer:
    """
    Bayesian optimizer for reward function hyperparameters.
    
    This class uses Bayesian optimization to find the optimal weights
    and parameters for reward components.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]], 
                 eval_function: Callable[[Dict[str, float]], float],
                 direction: str = "maximize",
                 n_trials: int = 50,
                 study_name: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) tuples
            eval_function: Function that takes parameters and returns a scalar to optimize
            direction: "maximize" or "minimize"
            n_trials: Number of optimization trials to run
            study_name: Name for the optuna study
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("The 'optuna' package is required for BayesianRewardOptimizer. "
                              "Install it with 'pip install optuna'")
            
        self.param_space = param_space
        self.eval_function = eval_function
        self.direction = direction
        self.n_trials = n_trials
        
        # Set up study name
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"reward_opt_{timestamp}"
        self.study_name = study_name
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _create_objective(self) -> Callable:
        """Create the objective function for optimization"""
        def objective(trial):
            # Sample parameters from parameter space
            params = {}
            for name, (low, high) in self.param_space.items():
                params[name] = trial.suggest_float(name, low, high)
                
            # Evaluate parameters
            return self.eval_function(params)
        
        return objective
    
    def optimize(self, show_progress_bar: bool = False) -> OptimizerResults:
        """
        Run the optimization process.
        
        Args:
            show_progress_bar: Whether to show a progress bar during optimization
            
        Returns:
            OptimizerResults object containing best parameters and value
        """
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction
        )
        
        # Create objective
        objective = self._create_objective()
        
        # Run optimization with optional progress bar
        if show_progress_bar:
            study.optimize(objective, n_trials=self.n_trials, 
                           show_progress_bar=True)
        else:
            study.optimize(objective, n_trials=self.n_trials)
            
        # Get best parameters and value
        best_params = study.best_params
        best_value = study.best_value
        
        # Store the results
        self.best_results = best_params
        
        # Return our custom results object
        return OptimizerResults(best_params=best_params, best_value=best_value)
    
    def generate_config(self, 
                        output_path: str, 
                        base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a config file with optimized parameters.
        
        Args:
            output_path: Path to save the generated config
            base_config: Base configuration to extend with optimized parameters
            
        Returns:
            The generated config dictionary
        """
        # Start with base config or empty dict
        if base_config is None:
            config = {}
        else:
            config = base_config.copy()
            
        # Ensure shaping_config exists
        if "shaping_config" not in config:
            config["shaping_config"] = {}
            
        # Add optimized parameters to config
        best_params = getattr(self, "best_results", None)
        
        # If we don't have stored results, create a simple config with parameters
        if best_params is None:
            # Use dummy values (this is just for testing)
            self.logger.warning("No optimization results found, using dummy values")
            best_params = {k: (v[0] + v[1])/2 for k, v in self.param_space.items()}
        
        # Process parameters into config
        for name, value in best_params.items():
            if "__" in name:
                # Format: "ComponentName__param_name"
                component, param = name.split("__", 1)
                
                if component not in config["shaping_config"]:
                    config["shaping_config"][component] = {}
                    
                if isinstance(config["shaping_config"][component], dict):
                    config["shaping_config"][component][param] = value
                else:
                    config["shaping_config"][component] = {param: value}
            else:
                # Simple parameter
                config["shaping_config"][name] = value
        
        # Include optimization metadata
        if "metadata" not in config:
            config["metadata"] = {}
        
        config["metadata"]["optimization"] = {
            "timestamp": datetime.now().isoformat(),
            "direction": self.direction,
            "n_trials": self.n_trials,
            "optimized": True
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save config to file
        with open(output_path, "w") as f:
            yaml.dump(config, f)
            
        return config
