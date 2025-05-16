import optuna
from typing import Callable, Dict, Any, List, Union, Optional, Tuple
import numpy as np
import os
import joblib
from datetime import datetime

class BayesianRewardOptimizer:
    def __init__(
        self,
        objective_function: Callable[[optuna.trial.Trial, Dict[str, Any]], float],
        search_space: Dict[str, Callable[[optuna.trial.Trial], Union[float, int, str]]],
        n_trials: int = 100,
        study_name: str = "bayesian_reward_optimization",
        storage: str = None,
        direction: str = "maximize",
        load_if_exists: bool = True,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        callbacks: List[Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]] = None
    ):
        self.objective_function = objective_function
        self.search_space = search_space
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.load_if_exists = load_if_exists
        self.pruner = pruner or optuna.pruners.MedianPruner()
        self.sampler = sampler or optuna.samplers.TPESampler()
        self.callbacks = callbacks or []
        self.study = None
        self.best_trial_params = {}
        self.optimization_history = []
        self.param_importances = {}

    def _optuna_objective(self, trial: optuna.trial.Trial) -> float:
        current_params = {}
        for param_name, suggester_func in self.search_space.items():
            current_params[param_name] = suggester_func(trial)
        
        return self.objective_function(trial, current_params)

    def optimize(self, show_progress_bar: bool = True) -> optuna.Study:
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler
        )
        
        if show_progress_bar:
            from tqdm import tqdm
            with tqdm(total=self.n_trials) as pbar:
                self.study.optimize(
                    self._optuna_objective, 
                    n_trials=self.n_trials,
                    callbacks=[
                        lambda study, trial: pbar.update(1),
                        *self.callbacks
                    ]
                )
        else:
            self.study.optimize(
                self._optuna_objective, 
                n_trials=self.n_trials,
                callbacks=self.callbacks
            )
        
        print(f"Optimization Finished for study: {self.study_name}")
        print(f"Number of finished trials: {len(self.study.trials)}")
        
        best_trial = self.study.best_trial
        print(f"Best trial value: {best_trial.value}")
        print("Best parameters found:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
        self.best_trial_params = best_trial.params
        self._update_optimization_metrics()
            
        return self.study
    
    def _update_optimization_metrics(self):
        if not self.study:
            return
        
        self.optimization_history = [(t.number, t.value) for t in self.study.trials if t.value is not None]
        
        try:
            self.param_importances = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            print(f"Could not compute parameter importances: {e}")
            self.param_importances = {}

    def get_best_params(self) -> Dict[str, Any]:
        if self.study and self.study.best_trial:
            return self.study.best_trial.params
        else:
            print("Optimization has not been run or no successful trials.")
            return {}

    def get_best_value(self) -> float:
        if self.study and self.study.best_trial:
            return self.study.best_trial.value
        else:
            print("Optimization has not been run or no successful trials.")
            return float('-inf') if self.direction == "maximize" else float('inf')
    
    def save_study(self, filepath: str = None) -> str:
        if not self.study:
            print("No study to save. Run optimize() first.")
            return None
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"rllama_study_{self.study_name}_{timestamp}.pkl"
        
        joblib.dump(self.study, filepath)
        print(f"Study saved to {filepath}")
        return filepath
    
    @classmethod
    def load_study(cls, filepath: str) -> 'BayesianRewardOptimizer':
        study = joblib.load(filepath)
        
        optimizer = cls(
            objective_function=lambda trial, params: 0,  
            search_space={},
            study_name=study.study_name,
            direction="maximize" if study.direction == optuna.study.StudyDirection.MAXIMIZE else "minimize"
        )
        
        optimizer.study = study
        optimizer._update_optimization_metrics()
        
        return optimizer
    
    def visualize(self, output_dir: str = None) -> Dict[str, Any]:
        if not self.study:
            print("No study to visualize. Run optimize() first.")
            return {}
        
        try:
            from rllama.rewards.visualization import RewardOptimizationVisualizer
            visualizer = RewardOptimizationVisualizer(self.study)
            return visualizer.plot_all(output_dir)
        except ImportError:
            print("Visualization module not found. Make sure matplotlib is installed.")
            return {}
    
    def suggest_next_params(self) -> Dict[str, Any]:
        if not self.study:
            print("No study available. Run optimize() first.")
            return {}
        
        next_trial = self.study.ask()
        next_params = {}
        
        for param_name, suggester_func in self.search_space.items():
            try:
                next_params[param_name] = suggester_func(next_trial)
            except Exception as e:
                print(f"Error suggesting parameter {param_name}: {e}")
        
        self.study.tell(next_trial, state=optuna.trial.TrialState.PRUNED)
        return next_params
    
    def evaluate_params(self, params: Dict[str, Any]) -> float:
        if not self.study:
            print("No study available. Run optimize() first.")
            return None
        
        trial = self.study.ask()
        
        try:
            value = self.objective_function(trial, params)
            self.study.tell(trial, value)
            return value
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return None
    
    def get_param_statistics(self) -> Dict[str, Dict[str, float]]:
        if not self.study:
            return {}
        
        df = self.study.trials_dataframe()
        
        stats = {}
        for param in self.search_space.keys():
            param_col = f'params_{param}'
            if param_col in df.columns:
                stats[param] = {
                    'mean': df[param_col].mean(),
                    'std': df[param_col].std(),
                    'min': df[param_col].min(),
                    'max': df[param_col].max(),
                    'median': df[param_col].median()
                }
        
        return stats

if __name__ == "__main__":
    
    class BaseRewardComponent:
        def __init__(self, name: str):
            self.name = name
            self.params = {}

        def configure(self, params: Dict[str, Any]):
            for key, value in params.items():
                if key in self.params:
                    self.params[key] = value
                elif hasattr(self, key):
                    setattr(self, key, value)

        def calculate(self, state: Any, action: Any) -> float:
            raise NotImplementedError

    class LengthReward(BaseRewardComponent):
        def __init__(self, weight: float = 1.0, target_length: int = 10):
            super().__init__("LengthReward")
            self.weight = weight
            self.target_length = target_length
            self.params = {"weight": self.weight, "target_length": self.target_length}

        def calculate(self, state: Any, action: Any) -> float:
            text_length = state.get('text_length', 0)
            return -self.weight * abs(text_length - self.target_length)

    class SpecificityReward(BaseRewardComponent):
        def __init__(self, weight: float = 1.0, desired_keywords: List[str] = None):
            super().__init__("SpecificityReward")
            self.weight = weight
            self.desired_keywords = desired_keywords if desired_keywords else []
            self.params = {"weight": self.weight} 

        def calculate(self, state: Any, action: Any) -> float:
            generated_text = state.get('text', '')
            score = 0
            for keyword in self.desired_keywords:
                if keyword in generated_text:
                    score += 1
            return self.weight * score

    reward_component_config = {
        "LengthReward": {
            "target_length": 15 
        },
        "SpecificityReward": {
            "desired_keywords": ["RL", "AI", "optimization"]
        }
    }

    def example_rl_objective_function(trial: optuna.trial.Trial, suggested_params: Dict[str, Any]) -> float:
        print(f"\nTrial {trial.number} evaluating with parameters: {suggested_params}")

        length_reward_comp = LengthReward(
            weight=suggested_params["LengthReward_weight"],
            target_length=reward_component_config["LengthReward"]["target_length"] 
        )
        
        specificity_reward_comp = SpecificityReward(
            weight=suggested_params["SpecificityReward_weight"],
            desired_keywords=reward_component_config["SpecificityReward"]["desired_keywords"]
        )
        
        simulated_text_length = trial.suggest_int("sim_text_length", 5, 25) 
        
        simulated_generated_texts = [
            "RL is a part of AI.",
            "Deep learning and optimization are key.",
            "This is a short text.",
            "AI and RL can be used for optimization problems."
        ]
        simulated_text = simulated_generated_texts[trial.number % len(simulated_generated_texts)]

        total_reward = 0
        num_eval_steps = 5 
        for i in range(num_eval_steps):
            current_text_length = simulated_text_length + (i - num_eval_steps // 2) 
            
            state = {
                'text_length': current_text_length,
                'text': simulated_text
            }
            
            len_r = length_reward_comp.calculate(state, None)
            spec_r = specificity_reward_comp.calculate(state, None)
            
            step_reward = len_r + spec_r
            total_reward += step_reward
            
            print(f"  Step {i}: TextLength={current_text_length}, LengthR={len_r:.2f}, SpecR={spec_r:.2f}, StepR={step_reward:.2f}")

        mean_reward = total_reward / num_eval_steps
        print(f"Trial {trial.number} mean_reward: {mean_reward:.2f}")
        
        return mean_reward

    search_space_definition = {
        "LengthReward_weight": lambda t: t.suggest_float("LengthReward_weight", 0.1, 2.0),
        "SpecificityReward_weight": lambda t: t.suggest_float("SpecificityReward_weight", 0.1, 3.0),
    }

    optimizer = BayesianRewardOptimizer(
        objective_function=example_rl_objective_function,
        search_space=search_space_definition,
        n_trials=10, 
        study_name="example_reward_tuning",
        storage="sqlite:///example_rllama_opt.db", 
        direction="maximize",
        load_if_exists=False,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print("Starting optimization...")
    study_results = optimizer.optimize(show_progress_bar=True)

    print("\n--- Optimization Summary ---")
    print(f"Best value: {optimizer.get_best_value()}")
    print(f"Best parameters: {optimizer.get_best_params()}")

    print("\n--- Parameter Statistics ---")
    stats = optimizer.get_param_statistics()
    for param, param_stats in stats.items():
        print(f"{param}:")
        for stat_name, stat_value in param_stats.items():
            print(f"  {stat_name}: {stat_value:.4f}")

    print("\n--- Saving Study ---")
    save_path = optimizer.save_study()

    print("\n--- Visualizing Results ---")
    output_dir = "optimization_results"
    figures = optimizer.visualize(output_dir)
    print(f"Visualization saved to {output_dir}")

    print("\n--- Loading Study ---")
    loaded_optimizer = BayesianRewardOptimizer.load_study(save_path)
    print(f"Loaded study with {len(loaded_optimizer.study.trials)} trials")