#!/usr/bin/env python3
"""
Advanced reward optimization module for RLlama.
Provides sophisticated optimization strategies for reward function hyperparameters.
"""

import os
import yaml
import numpy as np
import json
import time
from typing import Dict, Any, Callable, Optional, Union, List, Tuple, Protocol
from datetime import datetime
import logging
from pathlib import Path
from collections import defaultdict, deque
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class OptimizationResults:
    """Container for optimization results with comprehensive statistics."""
    
    def __init__(self, 
                 best_params: Dict[str, float], 
                 best_value: float,
                 optimization_history: List[Dict[str, Any]] = None,
                 convergence_info: Dict[str, Any] = None,
                 runtime_stats: Dict[str, float] = None):
        """
        Initialize optimization results.
        
        Args:
            best_params: Best parameters found
            best_value: Best objective value achieved
            optimization_history: History of all trials
            convergence_info: Information about convergence
            runtime_stats: Runtime statistics
        """
        self.best_params = best_params
        self.best_value = best_value
        self.optimization_history = optimization_history or []
        self.convergence_info = convergence_info or {}
        self.runtime_stats = runtime_stats or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary format."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'optimization_history': self.optimization_history,
            'convergence_info': self.convergence_info,
            'runtime_stats': self.runtime_stats
        }
        
    def save(self, filepath: str) -> None:
        """Save results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class EvaluationFunction(Protocol):
    """Protocol for evaluation functions."""
    
    def __call__(self, params: Dict[str, float]) -> float:
        """Evaluate parameters and return objective value."""
        ...

class BaseRewardOptimizer:
    """Base class for reward optimizers with common functionality."""
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]], 
                 eval_function: EvaluationFunction,
                 direction: str = "maximize",
                 random_seed: Optional[int] = None):
        """
        Initialize base optimizer.
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) tuples
            eval_function: Function that takes parameters and returns objective value
            direction: "maximize" or "minimize"
            random_seed: Random seed for reproducibility
        """
        self.param_space = param_space
        self.eval_function = eval_function
        self.direction = direction
        self.random_seed = random_seed
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization history
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, float]] = None
        self.best_value: Optional[float] = None
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def _evaluate_with_logging(self, params: Dict[str, float]) -> float:
        """Evaluate parameters with logging and error handling."""
        try:
            start_time = time.time()
            value = self.eval_function(params)
            evaluation_time = time.time() - start_time
            
            # Log evaluation
            self.history.append({
                'params': params.copy(),
                'value': value,
                'evaluation_time': evaluation_time,
                'timestamp': datetime.now().isoformat()
            })
            
            # Update best if needed
            if self.best_value is None or self._is_better(value, self.best_value):
                self.best_params = params.copy()
                self.best_value = value
                
            return value
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters {params}: {e}")
            # Return worst possible value
            return float('-inf') if self.direction == "maximize" else float('inf')
            
    def _is_better(self, value1: float, value2: float) -> bool:
        """Check if value1 is better than value2 based on direction."""
        if self.direction == "maximize":
            return value1 > value2
        else:
            return value1 < value2
            
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence information from optimization history."""
        if not self.history:
            return {}
            
        values = [entry['value'] for entry in self.history]
        
        # Calculate running best
        running_best = []
        current_best = values[0]
        
        for value in values:
            if self._is_better(value, current_best):
                current_best = value
            running_best.append(current_best)
            
        # Calculate improvement rate
        if len(running_best) > 10:
            recent_improvement = abs(running_best[-1] - running_best[-10])
            early_improvement = abs(running_best[9] - running_best[0]) if len(running_best) > 9 else 0
            improvement_rate = recent_improvement / max(early_improvement, 1e-8)
        else:
            improvement_rate = 1.0
            
        return {
            'total_evaluations': len(self.history),
            'best_value': self.best_value,
            'final_value': values[-1],
            'improvement_rate': improvement_rate,
            'convergence_trend': running_best[-10:] if len(running_best) >= 10 else running_best
        }

class BayesianRewardOptimizer(BaseRewardOptimizer):
    """
    Bayesian optimizer for reward function hyperparameters using Optuna.
    Provides sophisticated optimization with multiple sampling strategies.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]], 
                 eval_function: EvaluationFunction,
                 direction: str = "maximize",
                 n_trials: int = 100,
                 sampler: str = "tpe",
                 pruner: Optional[str] = "median",
                 study_name: Optional[str] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize Bayesian optimizer.
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) tuples
            eval_function: Function that takes parameters and returns objective value
            direction: "maximize" or "minimize"
            n_trials: Number of optimization trials
            sampler: Sampling strategy ("tpe", "cmaes", "random")
            pruner: Pruning strategy ("median", "successive_halving", None)
            study_name: Name for the optuna study
            random_seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for BayesianRewardOptimizer. "
                              "Install it with 'pip install optuna'")
                              
        super().__init__(param_space, eval_function, direction, random_seed)
        
        self.n_trials = n_trials
        self.sampler_name = sampler
        self.pruner_name = pruner
        
        # Set up study name
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"reward_opt_{timestamp}"
        self.study_name = study_name
        
        # Create sampler
        self.sampler = self._create_sampler()
        
        # Create pruner
        self.pruner = self._create_pruner()
        
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create appropriate sampler based on configuration."""
        if self.sampler_name == "tpe":
            return TPESampler(seed=self.random_seed)
        elif self.sampler_name == "cmaes":
            return CmaEsSampler(seed=self.random_seed)
        elif self.sampler_name == "random":
            return RandomSampler(seed=self.random_seed)
        else:
            self.logger.warning(f"Unknown sampler {self.sampler_name}, using TPE")
            return TPESampler(seed=self.random_seed)
            
    def _create_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create appropriate pruner based on configuration."""
        if self.pruner_name == "median":
            return MedianPruner()
        elif self.pruner_name == "successive_halving":
            return SuccessiveHalvingPruner()
        else:
            return None
            
    def _create_objective(self) -> Callable:
        """Create objective function for Optuna."""
        def objective(trial):
            # Sample parameters from parameter space
            params = {}
            for name, (low, high) in self.param_space.items():
                params[name] = trial.suggest_float(name, low, high)
                
            # Evaluate parameters
            return self._evaluate_with_logging(params)
        
        return objective
    
    def optimize(self, 
                 show_progress_bar: bool = True,
                 callbacks: Optional[List[Callable]] = None) -> OptimizationResults:
        """
        Run Bayesian optimization.
        
        Args:
            show_progress_bar: Whether to show progress bar
            callbacks: Optional list of callback functions
            
        Returns:
            OptimizationResults with comprehensive information
        """
        start_time = time.time()
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        # Create objective
        objective = self._create_objective()
        
        # Run optimization
        if show_progress_bar:
            study.optimize(objective, n_trials=self.n_trials, 
                           show_progress_bar=True, callbacks=callbacks)
        else:
            study.optimize(objective, n_trials=self.n_trials, callbacks=callbacks)
            
        # Calculate runtime statistics
        total_time = time.time() - start_time
        avg_time_per_trial = total_time / len(study.trials)
        
        runtime_stats = {
            'total_time': total_time,
            'avg_time_per_trial': avg_time_per_trial,
            'successful_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }
        
        # Get convergence info
        convergence_info = self.get_convergence_info()
        convergence_info['optuna_study_info'] = {
            'best_trial_number': study.best_trial.number,
            'total_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        }
        
        return OptimizationResults(
            best_params=study.best_params,
            best_value=study.best_value,
            optimization_history=self.history,
            convergence_info=convergence_info,
            runtime_stats=runtime_stats
        )

class GaussianProcessOptimizer(BaseRewardOptimizer):
    """
    Gaussian Process-based optimizer using scikit-learn.
    Good for expensive function evaluations with uncertainty quantification.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]], 
                 eval_function: EvaluationFunction,
                 direction: str = "maximize",
                 n_trials: int = 50,
                 n_initial_points: int = 10,
                 acquisition_function: str = "ei",
                 kernel: str = "matern",
                 random_seed: Optional[int] = None):
        """
        Initialize Gaussian Process optimizer.
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) tuples
            eval_function: Function that takes parameters and returns objective value
            direction: "maximize" or "minimize"
            n_trials: Total number of optimization trials
            n_initial_points: Number of random initial points
            acquisition_function: Acquisition function ("ei", "pi", "ucb")
            kernel: Kernel type ("matern", "rbf")
            random_seed: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for GaussianProcessOptimizer. "
                              "Install it with 'pip install scikit-learn'")
                              
        super().__init__(param_space, eval_function, direction, random_seed)
        
        self.n_trials = n_trials
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.kernel_name = kernel
        
        # Create kernel
        self.kernel = self._create_kernel()
        
        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            random_state=random_seed,
            normalize_y=True,
            alpha=1e-6
        )
        
    def _create_kernel(self):
        """Create appropriate kernel."""
        if self.kernel_name == "matern":
            return Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        elif self.kernel_name == "rbf":
            return RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
        else:
            return Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
            
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameter dict to array."""
        param_names = sorted(self.param_space.keys())
        return np.array([params[name] for name in param_names])
        
    def _array_to_params(self, array: np.ndarray) -> Dict[str, float]:
        """Convert array to parameter dict."""
        param_names = sorted(self.param_space.keys())
        return {name: float(array[i]) for i, name in enumerate(param_names)}
        
    def _sample_random_point(self) -> Dict[str, float]:
        """Sample a random point from parameter space."""
        params = {}
        for name, (low, high) in self.param_space.items():
            params[name] = np.random.uniform(low, high)
        return params
        
    def _acquisition_ei(self, X: np.ndarray, gp: GaussianProcessRegressor, 
                       best_value: float, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        if self.direction == "maximize":
            improvement = mu - best_value - xi
        else:
            improvement = best_value - mu - xi
            
        Z = improvement / sigma
        ei = improvement * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
        return ei.flatten()
        
    def _acquisition_pi(self, X: np.ndarray, gp: GaussianProcessRegressor, 
                       best_value: float, xi: float = 0.01) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        
        if self.direction == "maximize":
            improvement = mu - best_value - xi
        else:
            improvement = best_value - mu - xi
            
        Z = improvement / sigma
        return self._normal_cdf(Z)
        
    def _acquisition_ucb(self, X: np.ndarray, gp: GaussianProcessRegressor, 
                        kappa: float = 2.576) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        mu, sigma = gp.predict(X, return_std=True)
        
        if self.direction == "maximize":
            return mu + kappa * sigma
        else:
            return mu - kappa * sigma
            
    def _normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
    def _normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
    def optimize(self, show_progress: bool = True) -> OptimizationResults:
        """
        Run Gaussian Process optimization.
        
        Args:
            show_progress: Whether to show progress
            
        Returns:
            OptimizationResults with comprehensive information
        """
        start_time = time.time()
        
        # Phase 1: Random exploration
        if show_progress:
            print(f"Phase 1: Random exploration ({self.n_initial_points} points)")
            
        X_observed = []
        y_observed = []
        
        for i in range(self.n_initial_points):
            params = self._sample_random_point()
            value = self._evaluate_with_logging(params)
            
            X_observed.append(self._params_to_array(params))
            y_observed.append(value)
            
            if show_progress and (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{self.n_initial_points} initial points")
                
        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)
        
        # Phase 2: Gaussian Process guided optimization
        remaining_trials = self.n_trials - self.n_initial_points
        
        if show_progress:
            print(f"Phase 2: GP-guided optimization ({remaining_trials} points)")
            
        for i in range(remaining_trials):
            # Fit GP to observed data
            self.gp.fit(X_observed, y_observed)
            
            # Optimize acquisition function
            best_acquisition = float('-inf')
            best_candidate = None
            
            # Multi-start optimization of acquisition function
            for _ in range(100):  # Multiple random starts
                # Random starting point
                x0 = np.array([np.random.uniform(bounds[0], bounds[1]) 
                              for bounds in self.param_space.values()])
                
                # Define bounds for optimization
                bounds = list(self.param_space.values())
                
                # Optimize acquisition function
                if SCIPY_AVAILABLE:
                    def neg_acquisition(x):
                        X_test = x.reshape(1, -1)
                        if self.acquisition_function == "ei":
                            return -self._acquisition_ei(X_test, self.gp, self.best_value)[0]
                        elif self.acquisition_function == "pi":
                            return -self._acquisition_pi(X_test, self.gp, self.best_value)[0]
                        else:  # ucb
                            return -self._acquisition_ucb(X_test, self.gp)[0]
                    
                    result = minimize(neg_acquisition, x0, bounds=bounds, method='L-BFGS-B')
                    
                    if result.success:
                        acquisition_value = -result.fun
                        if acquisition_value > best_acquisition:
                            best_acquisition = acquisition_value
                            best_candidate = result.x
                            
            # Evaluate best candidate
            if best_candidate is not None:
                params = self._array_to_params(best_candidate)
            else:
                # Fallback to random sampling
                params = self._sample_random_point()
                
            value = self._evaluate_with_logging(params)
            
            # Add to observed data
            X_observed = np.vstack([X_observed, self._params_to_array(params)])
            y_observed = np.append(y_observed, value)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{remaining_trials} GP-guided points")
                
        # Calculate runtime statistics
        total_time = time.time() - start_time
        
        runtime_stats = {
            'total_time': total_time,
            'avg_time_per_trial': total_time / self.n_trials,
            'initial_exploration_time': total_time * (self.n_initial_points / self.n_trials)
        }
        
        return OptimizationResults(
            best_params=self.best_params,
            best_value=self.best_value,
            optimization_history=self.history,
            convergence_info=self.get_convergence_info(),
            runtime_stats=runtime_stats
        )

class MultiObjectiveOptimizer(BaseRewardOptimizer):
    """
    Multi-objective optimizer for reward functions with multiple competing objectives.
    Uses NSGA-II style optimization to find Pareto-optimal solutions.
    """
    
    def __init__(self, 
                 param_space: Dict[str, Tuple[float, float]], 
                 eval_functions: List[EvaluationFunction],
                 objective_names: List[str],
                 n_trials: int = 100,
                 population_size: int = 50,
                 random_seed: Optional[int] = None):
        """
        Initialize multi-objective optimizer.
        
        Args:
            param_space: Dictionary mapping parameter names to (min, max) tuples
            eval_functions: List of evaluation functions for each objective
            objective_names: Names of objectives
            n_trials: Number of optimization trials
            population_size: Size of population for genetic algorithm
            random_seed: Random seed for reproducibility
        """
        # Use first objective for base class
        super().__init__(param_space, eval_functions[0], "maximize", random_seed)
        
        self.eval_functions = eval_functions
        self.objective_names = objective_names
        self.n_trials = n_trials
        self.population_size = population_size
        
        # Pareto front tracking
        self.pareto_front: List[Dict[str, Any]] = []
        
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)."""
        at_least_one_better = False
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False
            elif v1 > v2:
                at_least_one_better = True
        return at_least_one_better
        
    def _evaluate_multi_objective(self, params: Dict[str, float]) -> List[float]:
        """Evaluate all objectives for given parameters."""
        objectives = []
        for eval_func in self.eval_functions:
            try:
                value = eval_func(params)
                objectives.append(value)
            except Exception as e:
                self.logger.error(f"Error in multi-objective evaluation: {e}")
                objectives.append(float('-inf'))
        return objectives
        
    def _update_pareto_front(self, params: Dict[str, float], objectives: List[float]) -> None:
        """Update Pareto front with new solution."""
        # Check if new solution is dominated by any existing solution
        dominated = False
        for existing in self.pareto_front:
            if self._dominates(existing['objectives'], objectives):
                dominated = True
                break
                
        if not dominated:
            # Remove any existing solutions dominated by new solution
            self.pareto_front = [
                existing for existing in self.pareto_front
                if not self._dominates(objectives, existing['objectives'])
            ]
            
            # Add new solution to front
            self.pareto_front.append({
                'params': params.copy(),
                'objectives': objectives.copy()
            })
            
    def optimize(self, show_progress: bool = True) -> OptimizationResults:
        """
        Run multi-objective optimization.
        
        Args:
            show_progress: Whether to show progress
            
        Returns:
            OptimizationResults with Pareto front information
        """
        start_time = time.time()
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            params = {name: np.random.uniform(bounds[0], bounds[1]) 
                     for name, bounds in self.param_space.items()}
            objectives = self._evaluate_multi_objective(params)
            
            population.append({
                'params': params,
                'objectives': objectives
            })
            
            self._update_pareto_front(params, objectives)
            
        if show_progress:
            print(f"Initialized population of {self.population_size}")
            
        # Evolution loop
        generations = self.n_trials // self.population_size
        
        for gen in range(generations):
            # Selection, crossover, mutation (simplified)
            new_population = []
            
            for _ in range(self.population_size):
                # Tournament selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                child_params = self._crossover(parent1['params'], parent2['params'])
                
                # Mutation
                child_params = self._mutate(child_params)
                
                # Evaluate child
                child_objectives = self._evaluate_multi_objective(child_params)
                
                new_population.append({
                    'params': child_params,
                    'objectives': child_objectives
                })
                
                self._update_pareto_front(child_params, child_objectives)
                
            population = new_population
            
            if show_progress and (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}/{generations}, Pareto front size: {len(self.pareto_front)}")
                
        # Select best solution (closest to ideal point)
        if self.pareto_front:
            ideal_point = [max(sol['objectives'][i] for sol in self.pareto_front) 
                          for i in range(len(self.objective_names))]
            
            best_distance = float('inf')
            best_solution = self.pareto_front[0]
            
            for solution in self.pareto_front:
                distance = np.linalg.norm(np.array(solution['objectives']) - np.array(ideal_point))
                if distance < best_distance:
                    best_distance = distance
                    best_solution = solution
                    
            best_params = best_solution['params']
            best_value = np.mean(best_solution['objectives'])  # Average of objectives
        else:
            best_params = {}
            best_value = float('-inf')
            
        runtime_stats = {
            'total_time': time.time() - start_time,
            'generations': generations,
            'pareto_front_size': len(self.pareto_front)
        }
        
        convergence_info = {
            'pareto_front': self.pareto_front,
            'objective_names': self.objective_names,
            'ideal_point': ideal_point if self.pareto_front else None
        }
        
        return OptimizationResults(
            best_params=best_params,
            best_value=best_value,
            optimization_history=self.history,
            convergence_info=convergence_info,
            runtime_stats=runtime_stats
        )
        
    def _tournament_selection(self, population: List[Dict[str, Any]], tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection for genetic algorithm."""
        tournament = np.random.choice(population, tournament_size, replace=False)
        
        # Select based on dominance
        best = tournament[0]
        for candidate in tournament[1:]:
            if self._dominates(candidate['objectives'], best['objectives']):
                best = candidate
                
        return best
        
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Uniform crossover for parameter dictionaries."""
        child = {}
        for name in parent1.keys():
            if np.random.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]
        return child
        
    def _mutate(self, params: Dict[str, float], mutation_rate: float = 0.1) -> Dict[str, float]:
        """Gaussian mutation for parameters."""
        mutated = params.copy()
        
        for name, value in mutated.items():
            if np.random.random() < mutation_rate:
                bounds = self.param_space[name]
                # Gaussian mutation with 10% of range as std
                std = (bounds[1] - bounds[0]) * 0.1
                new_value = value + np.random.normal(0, std)
                # Clip to bounds
                mutated[name] = np.clip(new_value, bounds[0], bounds[1])
                
        return mutated

def create_optimizer(optimizer_type: str, 
                    param_space: Dict[str, Tuple[float, float]], 
                    eval_function: EvaluationFunction,
                    **kwargs) -> BaseRewardOptimizer:
    """
    Factory function to create optimizers.
    
    Args:
        optimizer_type: Type of optimizer ("bayesian", "gp", "multi_objective")
        param_space: Parameter space definition
        eval_function: Evaluation function
        **kwargs: Additional arguments for optimizer
        
    Returns:
        Configured optimizer instance
    """
    if optimizer_type == "bayesian":
        return BayesianRewardOptimizer(param_space, eval_function, **kwargs)
    elif optimizer_type == "gp":
        return GaussianProcessOptimizer(param_space, eval_function, **kwargs)
    elif optimizer_type == "multi_objective":
        return MultiObjectiveOptimizer(param_space, eval_function, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

# Legacy compatibility
class BayesianRewardOptimizer_Legacy:
    """Legacy optimizer class for backward compatibility."""
    
    def __init__(self, param_space, eval_function, direction="maximize", n_trials=50, study_name=None):
        self.optimizer = BayesianRewardOptimizer(
            param_space=param_space,
            eval_function=eval_function,
            direction=direction,
            n_trials=n_trials,
            study_name=study_name
        )
        
    def optimize(self, show_progress_bar=False):
        results = self.optimizer.optimize(show_progress_bar=show_progress_bar)
        
        # Return legacy format
        class LegacyResults:
            def __init__(self, best_params, best_value):
                self.best_params = best_params
                self.best_value = best_value
                
        return LegacyResults(results.best_params, results.best_value)
        
    def generate_config(self, output_path, base_config=None):
        return self.optimizer.generate_config(output_path, base_config)

# Alias for backward compatibility
BayesianRewardOptimizer_Original = BayesianRewardOptimizer_Legacy
