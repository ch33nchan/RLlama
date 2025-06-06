import optuna
import yaml
import numpy as np
from typing import Dict, List, Callable, Any, Optional
import logging
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from Bayesian optimization"""
    best_params: Dict[str, Any]
    best_value: float
    study: optuna.Study
    trials_df: Any  # pandas DataFrame
    optimization_history: List[Dict]

class BayesianRewardOptimizer:
    """
    Bayesian optimization for RLlama reward component weights and parameters
    """
    
    def __init__(self, 
                 config_path: str,
                 study_name: str = "rllama_optimization",
                 storage: Optional[str] = None):
        """
        Initialize the Bayesian optimizer
        
        Args:
            config_path: Path to RLlama configuration file
            study_name: Name for the optimization study
            storage: Optuna storage URL (e.g., sqlite:///optuna.db)
        """
        self.config_path = config_path
        self.study_name = study_name
        self.storage = storage
        
        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Initialize optimization history
        self.optimization_history = []
        
        # Create study
        if storage:
            self.study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                load_if_exists=True
            )
        else:
            self.study = optuna.create_study(direction="maximize")
    
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define the search space for optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of parameter suggestions
        """
        params = {}
        
        # Optimize component weights
        components = self.base_config.get('composer', {}).get('components', [])
        
        for i, component in enumerate(components):
            component_type = component.get('type', f'component_{i}')
            base_weight = component.get('weight', 1.0)
            
            # Weight optimization (log scale for better exploration)
            params[f'{component_type}_weight'] = trial.suggest_float(
                f'{component_type}_weight',
                low=0.01,
                high=base_weight * 3.0,
                log=True
            )
            
            # Parameter optimization for specific components
            if component_type == 'CoherenceReward':
                params[f'{component_type}_min_sentences'] = trial.suggest_int(
                    f'{component_type}_min_sentences', 1, 5
                )
                params[f'{component_type}_transition_bonus'] = trial.suggest_float(
                    f'{component_type}_transition_bonus', 0.0, 0.5
                )
            
            elif component_type == 'HelpfulnessReward':
                params[f'{component_type}_overlap_weight'] = trial.suggest_float(
                    f'{component_type}_overlap_weight', 0.3, 1.0
                )
                params[f'{component_type}_question_bonus'] = trial.suggest_float(
                    f'{component_type}_question_bonus', 0.0, 0.5
                )
            
            elif component_type == 'ConcisenessReward':
                params[f'{component_type}_optimal_min'] = trial.suggest_int(
                    f'{component_type}_optimal_min', 3, 10
                )
                params[f'{component_type}_optimal_max'] = trial.suggest_int(
                    f'{component_type}_optimal_max', 15, 50
                )
            
            elif component_type == 'LengthReward':
                params[f'{component_type}_optimal_length'] = trial.suggest_int(
                    f'{component_type}_optimal_length', 10, 50
                )
                params[f'{component_type}_tolerance'] = trial.suggest_float(
                    f'{component_type}_tolerance', 0.1, 0.5
                )
        
        # Optimize normalization method
        params['normalization_method'] = trial.suggest_categorical(
            'normalization_method', 
            ['none', 'standard', 'minmax', 'robust']
        )
        
        # Optimize scheduling parameters
        for i, component in enumerate(components):
            component_type = component.get('type', f'component_{i}')
            schedule = component.get('schedule', {})
            
            if schedule.get('type') == 'exponential_decay':
                params[f'{component_type}_decay_rate'] = trial.suggest_float(
                    f'{component_type}_decay_rate', 0.9, 0.99
                )
            elif schedule.get('type') == 'linear_decay':
                params[f'{component_type}_decay_steps'] = trial.suggest_int(
                    f'{component_type}_decay_steps', 20, 200
                )
        
        return params
    
    def create_config_from_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a configuration dictionary from optimization parameters
        
        Args:
            params: Parameter dictionary from optimization
            
        Returns:
            Complete configuration dictionary
        """
        config = self.base_config.copy()
        
        # Update component weights and parameters
        components = config.get('composer', {}).get('components', [])
        
        for i, component in enumerate(components):
            component_type = component.get('type', f'component_{i}')
            
            # Update weight
            if f'{component_type}_weight' in params:
                component['weight'] = params[f'{component_type}_weight']
            
            # Update component-specific parameters
            if 'params' not in component:
                component['params'] = {}
            
            if component_type == 'CoherenceReward':
                if f'{component_type}_min_sentences' in params:
                    component['params']['min_sentences'] = params[f'{component_type}_min_sentences']
                if f'{component_type}_transition_bonus' in params:
                    component['params']['transition_bonus'] = params[f'{component_type}_transition_bonus']
            
            elif component_type == 'HelpfulnessReward':
                if f'{component_type}_overlap_weight' in params:
                    component['params']['overlap_weight'] = params[f'{component_type}_overlap_weight']
                if f'{component_type}_question_bonus' in params:
                    component['params']['question_bonus'] = params[f'{component_type}_question_bonus']
            
            elif component_type == 'ConcisenessReward':
                if f'{component_type}_optimal_min' in params:
                    component['params']['optimal_min'] = params[f'{component_type}_optimal_min']
                if f'{component_type}_optimal_max' in params:
                    component['params']['optimal_max'] = params[f'{component_type}_optimal_max']
            
            elif component_type == 'LengthReward':
                if f'{component_type}_optimal_length' in params:
                    component['params']['optimal_length'] = params[f'{component_type}_optimal_length']
                if f'{component_type}_tolerance' in params:
                    component['params']['tolerance'] = params[f'{component_type}_tolerance']
            
            # Update scheduling parameters
            if 'schedule' in component:
                schedule = component['schedule']
                if schedule.get('type') == 'exponential_decay':
                    if f'{component_type}_decay_rate' in params:
                        schedule['decay_rate'] = params[f'{component_type}_decay_rate']
                elif schedule.get('type') == 'linear_decay':
                    if f'{component_type}_decay_steps' in params:
                        schedule['decay_steps'] = params[f'{component_type}_decay_steps']
        
        # Update normalization method
        if 'normalization_method' in params:
            if 'shaper' not in config:
                config['shaper'] = {}
            config['shaper']['normalization_method'] = params['normalization_method']
        
        return config
    
    def objective_function(self, 
                          trial: optuna.Trial,
                          training_function: Callable,
                          evaluation_metrics: List[str] = None) -> float:
        """
        Objective function for optimization
        
        Args:
            trial: Optuna trial object
            training_function: Function that runs training and returns metrics
            evaluation_metrics: List of metrics to optimize for
            
        Returns:
            Objective value to maximize
        """
        # Get parameter suggestions
        params = self.define_search_space(trial)
        
        # Create configuration
        config = self.create_config_from_params(params)
        
        # Save temporary config
        temp_config_path = f"temp_config_trial_{trial.number}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Run training with this configuration
            metrics = training_function(temp_config_path)
            
            # Calculate objective value
            if evaluation_metrics is None:
                evaluation_metrics = ['mean_reward']
            
            objective_value = 0.0
            for metric in evaluation_metrics:
                if metric in metrics:
                    if metric == 'mean_reward':
                        objective_value += metrics[metric]
                    elif metric == 'reward_stability':
                        # Penalize high variance
                        objective_value -= metrics.get('reward_std', 0) * 0.1
                    elif metric == 'convergence_speed':
                        # Reward faster convergence
                        objective_value += 1.0 / max(metrics.get('steps_to_converge', 100), 1)
            
            # Store trial information
            trial_info = {
                'trial_number': trial.number,
                'params': params,
                'metrics': metrics,
                'objective_value': objective_value
            }
            self.optimization_history.append(trial_info)
            
            logger.info(f"Trial {trial.number}: Objective = {objective_value:.4f}")
            
            return objective_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -float('inf')
        
        finally:
            # Clean up temporary config
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    def optimize(self, 
                 training_function: Callable,
                 n_trials: int = 100,
                 evaluation_metrics: List[str] = None,
                 timeout: Optional[int] = None) -> OptimizationResult:
        """
        Run Bayesian optimization
        
        Args:
            training_function: Function that runs training given a config path
            n_trials: Number of optimization trials
            evaluation_metrics: Metrics to optimize for
            timeout: Timeout in seconds
            
        Returns:
            OptimizationResult object
        """
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        
        # Define the objective
        def objective(trial):
            return self.objective_function(trial, training_function, evaluation_metrics)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._trial_callback]
        )
        
        # Get results
        best_params = self.study.best_params
        best_value = self.study.best_value
        trials_df = self.study.trials_dataframe()
        
        logger.info(f"Optimization completed. Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            study=self.study,
            trials_df=trials_df,
            optimization_history=self.optimization_history
        )
    
    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function called after each trial"""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(f"Trial {trial.number} completed with value: {trial.value:.4f}")
        elif trial.state == optuna.trial.TrialState.FAIL:
            logger.warning(f"Trial {trial.number} failed")
    
    def save_best_config(self, output_path: str):
        """Save the best configuration to a file"""
        if not hasattr(self.study, 'best_params'):
            raise ValueError("No optimization has been run yet")
        
        best_config = self.create_config_from_params(self.study.best_params)
        
        with open(output_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        logger.info(f"Best configuration saved to {output_path}")
    
    def get_optimization_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization of optimization results"""
        if not self.optimization_history:
            return {}
        
        trials = []
        objectives = []
        params_history = {}
        
        for trial_info in self.optimization_history:
            trials.append(trial_info['trial_number'])
            objectives.append(trial_info['objective_value'])
            
            for param_name, param_value in trial_info['params'].items():
                if param_name not in params_history:
                    params_history[param_name] = []
                params_history[param_name].append(param_value)
        
        return {
            'trials': trials,
            'objectives': objectives,
            'params_history': params_history,
            'best_trial': max(range(len(objectives)), key=lambda i: objectives[i]),
            'best_objective': max(objectives) if objectives else 0
        }