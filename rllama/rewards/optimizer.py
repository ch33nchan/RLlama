# rllama/rewards/optimizer.py

import optuna
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
import os
import json
import yaml
from functools import partial


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_params: Dict[str, Any]
    best_value: float
    study: optuna.study.Study
    param_importances: Dict[str, float]
    history: List[Dict[str, Any]]
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def save(self, filepath: str):
        """Save optimization results to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        result_dict = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "param_importances": self.param_importances,
            "history": self.history,
            "timestamp": self.timestamp
        }
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(result_dict, f)
            else:
                json.dump(result_dict, f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'OptimizationResult':
        """Load optimization results from a file"""
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(
            best_params=data["best_params"],
            best_value=data["best_value"],
            study=None,  # Study object can't be serialized easily
            param_importances=data["param_importances"],
            history=data["history"],
            timestamp=data["timestamp"]
        )


class BayesianRewardOptimizer:
    """
    Optimizes reward function parameters using Bayesian optimization via Optuna.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Any],
                 eval_function: Callable[[Dict[str, Any]], float],
                 direction: str = "maximize",
                 n_trials: int = 50,
                 study_name: Optional[str] = None,
                 storage: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Args:
            param_space: Dictionary defining parameter space for optimization.
                Each key is a parameter name and value defines the range.
                Examples:
                {
                    "length_weight": (0.0, 1.0),          # Uniform float range
                    "toxicity_penalty": [-10.0, -5.0, -1.0],  # Discrete values
                    "model_name": ["gpt2", "gpt2-medium", "gpt2-large"]  # Categorical
                }
                
            eval_function: Function that takes parameter dictionary and returns a score.
                The function signature should be: func(params: Dict[str, Any]) -> float
                
            direction: Optimization direction, either "maximize" or "minimize".
            n_trials: Number of trials for optimization.
            study_name: Optional name for the study.
            storage: Optional storage URL for Optuna.
        """
        self.param_space = param_space
        self.eval_function = eval_function
        self.direction = direction
        self.n_trials = n_trials
        self.study_name = study_name or f"reward_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def _create_objective(self) -> Callable[[optuna.trial.Trial], float]:
        """Create the objective function for Optuna"""
        def objective(trial: optuna.trial.Trial) -> float:
            # Generate parameters based on the parameter space
            params = {}
            for param_name, param_range in self.param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Handle numeric ranges
                    if all(isinstance(v, int) for v in param_range):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Handle categorical or discrete values
                    if all(isinstance(v, (int, float)) for v in param_range):
                        params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range),
                                                              step=(param_range[1] - param_range[0]) if len(param_range) > 1 else None)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_range)
                else:
                    raise ValueError(f"Unsupported parameter range specification for {param_name}: {param_range}")
            
            # Evaluate the parameters
            return self.eval_function(params)
        
        return objective
    
    def optimize(self, show_progress_bar: bool = True) -> OptimizationResult:
        """
        Run the optimization process.
        
        Args:
            show_progress_bar: Whether to show a progress bar during optimization.
            
        Returns:
            OptimizationResult containing the best parameters and other information.
        """
        # Create or load a study
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True
        )
        
        # Run optimization
        self.study.optimize(
            self._create_objective(),
            n_trials=self.n_trials,
            show_progress_bar=show_progress_bar
        )
        
        # Store results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        # Get parameter importance if we have more than one parameter
        if len(self.param_space) > 1:
            try:
                param_importances = optuna.importance.get_param_importances(self.study)
            except Exception:
                param_importances = {param: 1.0 / len(self.param_space) for param in self.param_space}
        else:
            param_importances = {next(iter(self.param_space.keys())): 1.0}
        
        # Prepare history
        history = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime": trial.datetime.isoformat() if trial.datetime else None
                })
        
        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            study=self.study,
            param_importances=param_importances,
            history=history
        )
    
    def generate_config(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a configuration dictionary from the best parameters.
        
        Args:
            output_path: Optional path to save the configuration as YAML.
            
        Returns:
            The configuration dictionary.
        """
        if self.best_params is None:
            raise ValueError("No optimization results available. Run optimize() first.")
        
        # Create a config from the best parameters
        config = {"reward_components": []}
        
        # Group parameters by component
        component_params = {}
        for param_name, param_value in self.best_params.items():
            if "__" in param_name:
                component_name, param_key = param_name.split("__", 1)
                if component_name not in component_params:
                    component_params[component_name] = {}
                component_params[component_name][param_key] = param_value
            else:
                # Global parameters go into the root config
                config[param_name] = param_value
        
        # Create component configurations
        for component_name, params in component_params.items():
            component_config = {
                "name": component_name,
                "params": params
            }
            config["reward_components"].append(component_config)
        
        # Save config if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(config, f)
        
        return config

class GridSearchOptimizer:
    """
    Optimizes reward function parameters using grid search.
    Useful for exploring smaller parameter spaces exhaustively.
    """
    
    def __init__(self, 
                 param_grid: Dict[str, List[Any]],
                 eval_function: Callable[[Dict[str, Any]], float],
                 direction: str = "maximize"):
        """
        Initialize the grid search optimizer.
        
        Args:
            param_grid: Dictionary defining parameter grid for search.
                Each key is a parameter name and value is a list of values to try.
                
            eval_function: Function that takes parameter dictionary and returns a score.
                
            direction: Optimization direction, either "maximize" or "minimize".
        """
        self.param_grid = param_grid
        self.eval_function = eval_function
        self.direction = direction
        self.results = []
        self.best_params = None
        self.best_value = None
    
    def _generate_parameter_combinations(self):
        """Generate all combinations of parameters from the grid"""
        import itertools
        
        # Get all parameter names and their possible values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Generate all combinations
        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))
    
    def optimize(self, show_progress_bar: bool = True) -> OptimizationResult:
        """
        Run the grid search optimization process.
        
        Args:
            show_progress_bar: Whether to show a progress bar during optimization.
            
        Returns:
            OptimizationResult containing the best parameters and other information.
        """
        from tqdm import tqdm
        
        # Generate all parameter combinations
        combinations = list(self._generate_parameter_combinations())
        
        # Create a progress bar if requested
        if show_progress_bar:
            iterator = tqdm(combinations, desc="Grid Search")
        else:
            iterator = combinations
        
        # Evaluate all combinations
        for params in iterator:
            value = self.eval_function(params)
            self.results.append({
                "params": params,
                "value": value,
                "timestamp": datetime.now().isoformat()
            })
        
        # Sort results by value
        self.results.sort(key=lambda x: x["value"], reverse=(self.direction == "maximize"))
        
        # Get the best parameters
        self.best_params = self.results[0]["params"]
        self.best_value = self.results[0]["value"]
        
        # Calculate simple parameter importance
        param_importances = self._calculate_param_importance()
        
        # Create a fake study for compatibility with OptimizationResult
        study = optuna.create_study(direction=self.direction)
        for result in self.results:
            trial = optuna.trial.create_trial(
                params=result["params"],
                distributions={k: optuna.distributions.CategoricalDistribution([v]) 
                              for k, v in result["params"].items()},
                value=result["value"]
            )
            study.add_trial(trial)
        
        return OptimizationResult(
            best_params=self.best_params,
            best_value=self.best_value,
            study=study,
            param_importances=param_importances,
            history=self.results
        )
    
    def _calculate_param_importance(self) -> Dict[str, float]:
        """Calculate simple parameter importance based on variance of results"""
        if not self.results:
            return {param: 1.0 / len(self.param_grid) for param in self.param_grid}
        
        importances = {}
        for param_name in self.param_grid:
            # Group results by parameter value
            grouped_results = {}
            for result in self.results:
                param_value = result["params"][param_name]
                if param_value not in grouped_results:
                    grouped_results[param_value] = []
                grouped_results[param_value].append(result["value"])
            
            # Calculate variance between groups
            group_means = [sum(values) / len(values) for values in grouped_results.values()]
            if len(group_means) > 1:
                variance = np.var(group_means)
                importances[param_name] = variance
            else:
                importances[param_name] = 0.0
        
        # Normalize importances to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}
        else:
            importances = {k: 1.0 / len(importances) for k in importances}
        
        return importances