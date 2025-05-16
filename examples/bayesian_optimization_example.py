import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import optuna
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllama.rewards.optimizer import BayesianRewardOptimizer
from rllama.rewards.visualization import RewardOptimizationVisualizer, RewardComponentVisualizer
from rllama.rewards.composition import RewardComposer

class FactualityReward:
    def __init__(self, weight=1.0, threshold=0.7):
        self.name = "FactualityReward"
        self.weight = weight
        self.threshold = threshold
    
    def calculate(self, state, action=None):
        factuality_score = state.get('factuality_score', 0.5)
        if factuality_score < self.threshold:
            return -self.weight * (self.threshold - factuality_score) * 10
        else:
            return self.weight * (factuality_score - self.threshold)

class CoherenceReward:
    def __init__(self, weight=1.0, min_score=0.0, max_score=1.0):
        self.name = "CoherenceReward"
        self.weight = weight
        self.min_score = min_score
        self.max_score = max_score
    
    def calculate(self, state, action=None):
        coherence_score = state.get('coherence_score', 0.5)
        normalized_score = (coherence_score - self.min_score) / (self.max_score - self.min_score)
        return self.weight * normalized_score

class RelevanceReward:
    def __init__(self, weight=1.0, query_importance=0.5):
        self.name = "RelevanceReward"
        self.weight = weight
        self.query_importance = query_importance
    
    def calculate(self, state, action=None):
        relevance_score = state.get('relevance_score', 0.5)
        query_match = state.get('query_match', 0.5)
        
        combined_score = (1 - self.query_importance) * relevance_score + self.query_importance * query_match
        return self.weight * combined_score

def simulate_llm_response(params: Dict[str, float]) -> Dict[str, float]:
    factuality_weight = params.get("FactualityReward_weight", 1.0)
    coherence_weight = params.get("CoherenceReward_weight", 1.0)
    relevance_weight = params.get("RelevanceReward_weight", 1.0)
    factuality_threshold = params.get("FactualityReward_threshold", 0.7)
    
    base_factuality = 0.6 + 0.3 * (factuality_threshold / 0.7)
    base_coherence = 0.7 - 0.2 * (factuality_weight / coherence_weight) if coherence_weight > 0 else 0.5
    base_relevance = 0.8 - 0.1 * (factuality_weight / relevance_weight) if relevance_weight > 0 else 0.5
    
    noise = 0.1
    
    return {
        'factuality_score': min(1.0, max(0.0, base_factuality + noise * (np.random.random() - 0.5))),
        'coherence_score': min(1.0, max(0.0, base_coherence + noise * (np.random.random() - 0.5))),
        'relevance_score': min(1.0, max(0.0, base_relevance + noise * (np.random.random() - 0.5))),
        'query_match': min(1.0, max(0.0, 0.6 + noise * (np.random.random() - 0.5)))
    }

def objective_function(trial: optuna.trial.Trial, params: Dict[str, Any]) -> float:
    factuality_reward = FactualityReward(
        weight=params["FactualityReward_weight"],
        threshold=params["FactualityReward_threshold"]
    )
    
    coherence_reward = CoherenceReward(
        weight=params["CoherenceReward_weight"]
    )
    
    relevance_reward = RelevanceReward(
        weight=params["RelevanceReward_weight"],
        query_importance=params["RelevanceReward_query_importance"]
    )
    
    components = {
        "factuality": factuality_reward,
        "coherence": coherence_reward,
        "relevance": relevance_reward
    }
    
    composer = RewardComposer(components)
    
    total_reward = 0.0
    n_samples = 20
    
    component_visualizer = RewardComponentVisualizer()
    
    for i in range(n_samples):
        state = simulate_llm_response(params)
        
        component_rewards = {}
        for name, component in components.items():
            component_rewards[name] = component.calculate(state)
        
        reward = composer.calculate(state, None)
        total_reward += reward
        
        component_visualizer.record_reward(reward, component_rewards, i)
    
    mean_reward = total_reward / n_samples
    
    if trial.number < 5:
        component_visualizer.plot_component_contributions(f"trial_{trial.number}_components.png")
        component_visualizer.plot_reward_history(window_size=3, save_path=f"trial_{trial.number}_history.png")
    
    print(f"Trial {trial.number}: Mean reward = {mean_reward:.4f}")
    
    return mean_reward

def main():
    search_space = {
        "FactualityReward_weight": lambda t: t.suggest_float("FactualityReward_weight", 0.1, 2.0),
        "FactualityReward_threshold": lambda t: t.suggest_float("FactualityReward_threshold", 0.5, 0.9),
        "CoherenceReward_weight": lambda t: t.suggest_float("CoherenceReward_weight", 0.1, 2.0),
        "RelevanceReward_weight": lambda t: t.suggest_float("RelevanceReward_weight", 0.1, 2.0),
        "RelevanceReward_query_importance": lambda t: t.suggest_float("RelevanceReward_query_importance", 0.1, 0.9)
    }
    
    optimizer = BayesianRewardOptimizer(
        objective_function=objective_function,
        search_space=search_space,
        n_trials=30,
        study_name="llm_reward_optimization",
        storage="sqlite:///llm_reward_optimization.db",
        direction="maximize",
        load_if_exists=False
    )
    
    print("Starting optimization...")
    study = optimizer.optimize()
    
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\nGenerating visualization...")
    visualizer = RewardOptimizationVisualizer(study)
    visualizer.create_dashboard("optimization_dashboard.png")
    
    print("Visualizing reward landscape...")
    
    best_params = study.best_params
    param_names = list(best_params.keys())
    
    if len(param_names) >= 2:
        x_param = "FactualityReward_weight"
        y_param = "CoherenceReward_weight"
        
        x_range = np.linspace(0.1, 2.0, 20)
        y_range = np.linspace(0.1, 2.0, 20)
        
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                test_params = best_params.copy()
                test_params[x_param] = X[j, i]
                test_params[y_param] = Y[j, i]
                
                state = simulate_llm_response(test_params)
                
                factuality_reward = FactualityReward(
                    weight=test_params["FactualityReward_weight"],
                    threshold=test_params["FactualityReward_threshold"]
                )
                
                coherence_reward = CoherenceReward(
                    weight=test_params["CoherenceReward_weight"]
                )
                
                relevance_reward = RelevanceReward(
                    weight=test_params["RelevanceReward_weight"],
                    query_importance=test_params["RelevanceReward_query_importance"]
                )
                
                components = {
                    "factuality": factuality_reward,
                    "coherence": coherence_reward,
                    "relevance": relevance_reward
                }
                
                composer = RewardComposer(components)
                Z[j, i] = composer.calculate(state, None)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, 50, cmap='viridis')
        plt.colorbar(label='Reward Value')
        plt.scatter(best_params[x_param], best_params[y_param], color='red', marker='*', s=200, label='Best Parameters')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f'Reward Landscape for {x_param} vs {y_param}')
        plt.legend()
        plt.savefig("reward_landscape.png")
        plt.close()
    
    print("Visualizations saved as 'optimization_dashboard.png' and 'reward_landscape.png'")

if __name__ == "__main__":
    main()