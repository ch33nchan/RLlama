#!/usr/bin/env python3
"""
Complete Bayesian Optimization Cookbook
======================================

This cookbook demonstrates automated hyperparameter tuning for reward functions
using state-of-the-art Bayesian optimization with RLlama.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
from typing import Dict, Any, List, Tuple
import seaborn as sns

sys.path.append(os.path.abspath("../.."))

from rllama import RewardEngine
from rllama.rewards.optimizer import BayesianRewardOptimizer

def create_evaluation_scenarios():
    """Create diverse evaluation scenarios for testing reward functions"""
    scenarios = [
        {
            'name': 'Creative Writing',
            'responses': [
                "The dragon soared through the crimson sky, its scales shimmering like molten gold in the dying light.",
                "Dragon fly sky red.",
                "Once upon a time there was a dragon that could fly and it was in the sky and the sky was red.",
                "In the realm of Eldoria, the ancient wyrm Pyraxis emerged from slumber, his magnificent form casting shadows."
            ],
            'queries': ["Write a creative fantasy scene"] * 4,
            'target_scores': [0.9, 0.2, 0.4, 0.85]  # Expected quality scores
        },
        {
            'name': 'Technical Explanation',
            'responses': [
                "Machine learning is a subset of AI that enables computers to learn patterns from data without explicit programming.",
                "ML use computer learn data.",
                "Machine learning is when computers can learn things from data and then make predictions.",
                "ML algorithms utilize statistical techniques to identify patterns, enabling automated decision-making."
            ],
            'queries': ["Explain machine learning"] * 4,
            'target_scores': [0.9, 0.15, 0.6, 0.85]
        },
        {
            'name': 'Code Quality',
            'responses': [
                "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "fib = lambda n: n if n <= 1 else fib(n-1) + fib(n-2)",
                "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "def fibonacci(number):\n    \"\"\"Calculate nth Fibonacci.\"\"\"\n    if number <= 1:\n        return number\n    return fibonacci(number - 1) + fibonacci(number - 2)"
            ],
            'queries': ["Write a Fibonacci function"] * 4,
            'target_scores': [0.8, 0.7, 0.6, 0.95]
        }
    ]
    return scenarios

def create_base_reward_config():
    """Create a base reward configuration for optimization"""
    return {
        'reward_components': [
            {
                'name': 'LengthReward',
                'params': {
                    'target_length': 100,
                    'strength': 1.0,
                    'tolerance': 0.2
                }
            },
            {
                'name': 'DiversityReward', 
                'params': {
                    'history_size': 10,
                    'similarity_threshold': 0.8,
                    'diversity_weight': 1.0
                }
            },
            {
                'name': 'ConstantReward',
                'params': {
                    'value': 0.1
                }
            }
        ],
        'shaping_config': {
            'LengthReward': {'weight': 1.0},
            'DiversityReward': {'weight': 0.5},
            'ConstantReward': {'weight': 0.1}
        }
    }

def comprehensive_evaluation_function(params: Dict[str, float]) -> float:
    """
    Comprehensive evaluation function that tests reward parameters
    across multiple scenarios and metrics.
    """
    
    # Create temporary config with optimized parameters
    base_config = create_base_reward_config()
    
    # Update parameters
    for param_name, value in params.items():
        if '__' in param_name:
            component, param = param_name.split('__', 1)
            
            # Update component params
            for comp_config in base_config['reward_components']:
                if comp_config['name'] == component:
                    comp_config['params'][param] = value
                    break
            
            # Update shaping weights
            if param == 'weight':
                base_config['shaping_config'][component] = {'weight': value}
    
    # Save temporary config
    temp_config_path = './temp_optimization_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(base_config, f)
    
    try:
        # Initialize reward engine
        engine = RewardEngine(temp_config_path)
        
        # Load evaluation scenarios
        scenarios = create_evaluation_scenarios()
        
        total_score = 0.0
        total_weight = 0.0
        
        for scenario in scenarios:
            scenario_scores = []
            
            for response, query, target_score in zip(
                scenario['responses'], 
                scenario['queries'], 
                scenario['target_scores']
            ):
                context = {
                    'response': response,
                    'query': query,
                    'metadata': {'scenario': scenario['name']}
                }
                
                # Compute reward
                reward = engine.compute(context)
                
                # Normalize reward to [0, 1] range
                normalized_reward = np.tanh(reward / 2.0) * 0.5 + 0.5
                
                # Calculate alignment with target score
                score_diff = abs(normalized_reward - target_score)
                quality_score = 1.0 - score_diff
                
                scenario_scores.append(quality_score)
            
            # Weight scenarios by importance
            scenario_weights = {
                'Creative Writing': 1.0,
                'Technical Explanation': 1.2,
                'Code Quality': 1.1
            }
            
            scenario_weight = scenario_weights.get(scenario['name'], 1.0)
            scenario_avg = np.mean(scenario_scores)
            
            total_score += scenario_avg * scenario_weight
            total_weight += scenario_weight
        
        # Add regularization terms
        
        # 1. Penalty for extreme parameter values
        param_penalty = 0.0
        for value in params.values():
            if abs(value) > 10.0:
                param_penalty += 0.1 * (abs(value) - 10.0)
        
        # 2. Reward for balanced component weights
        weight_params = [v for k, v in params.items() if 'weight' in k]
        if len(weight_params) > 1:
            weight_variance = np.var(weight_params)
            balance_bonus = 0.1 * np.exp(-weight_variance)
        else:
            balance_bonus = 0.0
        
        # Final score
        final_score = (total_score / total_weight) - param_penalty + balance_bonus
        
        return final_score
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

def run_single_objective_optimization():
    """Run single-objective Bayesian optimization"""
    print("\n🎯 Single-Objective Bayesian Optimization")
    print("-" * 40)
    
    # Define parameter space
    param_space = {
        'LengthReward__strength': (0.1, 3.0),
        'LengthReward__tolerance': (0.05, 0.5),
        'DiversityReward__diversity_weight': (0.1, 2.0),
        'LengthReward__weight': (0.1, 2.0),
        'DiversityReward__weight': (0.1, 2.0)
    }
    
    print(f"Parameter space: {len(param_space)} dimensions")
    print(f"Total search space volume: ~{np.prod([high-low for low, high in param_space.values()]):.2e}")
    
    # Create optimizer
    optimizer = BayesianRewardOptimizer(
        param_space=param_space,
        eval_function=comprehensive_evaluation_function,
        direction='maximize',
        n_trials=40
    )
    
    # Run optimization
    print("Starting Bayesian optimization...")
    start_time = time.time()
    results = optimizer.optimize(show_progress_bar=True)
    optimization_time = time.time() - start_time
    
    print(f"\n📊 Optimization Results:")
    print(f"Best score: {results.best_value:.6f}")
    print(f"Best parameters: {results.best_params}")
    print(f"Optimization time: {optimization_time:.2f}s")
    print(f"Trials completed: {len(results.trials)}")
    
    # Generate optimized configuration
    config_path = "./output/optimized_single_objective_config.yaml"
    optimizer.generate_config(config_path, create_base_reward_config())
    
    return results, optimization_time

def run_multi_objective_optimization():
    """Run multi-objective optimization"""
    print("\n🎯 Multi-Objective Optimization")
    print("-" * 40)
    
    def multi_objective_function(params):
        """Evaluate multiple objectives simultaneously"""
        base_config = create_base_reward_config()
        
        # Update parameters
        for param_name, value in params.items():
            if '__' in param_name:
                component, param = param_name.split('__', 1)
                for comp_config in base_config['reward_components']:
                    if comp_config['name'] == component:
                        comp_config['params'][param] = value
                        break
        
        # Save temporary config
        temp_config_path = './temp_multi_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(base_config, f)
        
        try:
            engine = RewardEngine(temp_config_path)
            
            # Evaluate multiple objectives
            objectives = {
                'quality': 0.0,
                'diversity': 0.0,
                'consistency': 0.0
            }
            
            # Test scenarios for different objectives
            test_scenarios = [
                {'response': 'High quality technical explanation with detailed examples', 
                 'query': 'Explain neural networks'},
                {'response': 'Creative story with unique plot twists', 
                 'query': 'Write a creative story'},
                {'response': 'Short answer', 'query': 'What is AI?'},
                {'response': 'Medium length response with good balance', 
                 'query': 'Describe machine learning'}
            ]
            
            rewards = []
            for scenario in test_scenarios:
                reward = engine.compute(scenario)
                rewards.append(reward)
            
            # Calculate objectives
            objectives['quality'] = np.mean(rewards)
            objectives['diversity'] = np.std(rewards)
            objectives['consistency'] = 1.0 / (np.std(rewards) + 0.1)
            
            # Combined objective (weighted sum)
            combined = (
                0.5 * objectives['quality'] +
                0.3 * objectives['diversity'] +
                0.2 * objectives['consistency']
            )
            
            return combined
            
        finally:
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    # Run multi-objective optimization
    multi_param_space = {
        'LengthReward__strength': (0.5, 2.0),
        'LengthReward__tolerance': (0.1, 0.4),
        'DiversityReward__diversity_weight': (0.5, 1.5)
    }
    
    multi_optimizer = BayesianRewardOptimizer(
        param_space=multi_param_space,
        eval_function=multi_objective_function,
        direction='maximize',
        n_trials=25
    )
    
    multi_results = multi_optimizer.optimize(show_progress_bar=True)
    
    print(f"Multi-objective best score: {multi_results.best_value:.6f}")
    print(f"Multi-objective best params: {multi_results.best_params}")
    
    return multi_results

def parameter_sensitivity_analysis():
    """Perform parameter sensitivity analysis"""
    print("\n🔍 Parameter Sensitivity Analysis")
    print("-" * 40)
    
    # Define parameter ranges for sensitivity analysis
    sensitivity_params = {
        'LengthReward__strength': np.linspace(0.1, 3.0, 15),
        'LengthReward__tolerance': np.linspace(0.05, 0.5, 15),
        'DiversityReward__diversity_weight': np.linspace(0.1, 2.0, 15),
        'LengthReward__weight': np.linspace(0.1, 2.0, 15)
    }
    
    # Base parameters
    base_params = {
        'LengthReward__strength': 1.0,
        'LengthReward__tolerance': 0.2,
        'DiversityReward__diversity_weight': 1.0,
        'LengthReward__weight': 1.0,
        'DiversityReward__weight': 0.5
    }
    
    sensitivity_results = {}
    
    for param_name, param_values in sensitivity_params.items():
        print(f"  Analyzing sensitivity for {param_name}...")
        
        scores = []
        for value in param_values:
            # Create test parameters
            test_params = base_params.copy()
            test_params[param_name] = value
            
            # Evaluate this parameter setting
            try:
                score = comprehensive_evaluation_function(test_params)
                scores.append(score)
            except Exception as e:
                print(f"    Error evaluating {param_name}={value}: {e}")
                scores.append(0.0)
        
        sensitivity_results[param_name] = {
            'values': param_values,
            'scores': scores,
            'sensitivity': np.std(scores)  # Higher std = more sensitive
        }
        
        print(f"    Sensitivity score: {np.std(scores):.4f}")
    
    return sensitivity_results

def run_constrained_optimization():
    """Run optimization with constraints"""
    print("\n⚖️ Constrained Optimization")
    print("-" * 40)
    
    def constrained_function(params):
        """Evaluation function with constraints"""
        # Constraint: total weight budget
        weight_params = [v for k, v in params.items() if 'weight' in k]
        total_weight = sum(weight_params)
        
        # Penalty for exceeding weight budget
        weight_budget = 3.0
        if total_weight > weight_budget:
            penalty = -10.0 * (total_weight - weight_budget)
        else:
            penalty = 0.0
        
        # Base evaluation
        base_score = comprehensive_evaluation_function(params)
        
        return base_score + penalty
    
    constrained_param_space = {
        'LengthReward__weight': (0.1, 2.0),
        'DiversityReward__weight': (0.1, 2.0),
        'ConstantReward__weight': (0.01, 1.0)
    }
    
    constrained_optimizer = BayesianRewardOptimizer(
        param_space=constrained_param_space,
        eval_function=constrained_function,
        direction='maximize',
        n_trials=20
    )
    
    constrained_results = constrained_optimizer.optimize(show_progress_bar=True)
    
    # Check if constraint is satisfied
    weight_params = [v for k, v in constrained_results.best_params.items() if 'weight' in k]
    total_weight = sum(weight_params)
    
    print(f"Constrained optimization best score: {constrained_results.best_value:.6f}")
    print(f"Total weight: {total_weight:.3f} (budget: 3.0)")
    print(f"Constraint satisfied: {'✅' if total_weight <= 3.0 else '❌'}")
    
    return constrained_results

def create_optimization_visualizations(single_results, multi_results, sensitivity_results, constrained_results):
    """Create comprehensive optimization visualizations"""
    print("\n📊 Creating optimization visualizations...")
    
    os.makedirs("./output", exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Optimization convergence
    single_scores = [trial.value for trial in single_results.trials]
    multi_scores = [trial.value for trial in multi_results.trials]
    constrained_scores = [trial.value for trial in constrained_results.trials]
    
    axes[0, 0].plot(range(1, len(single_scores) + 1), np.maximum.accumulate(single_scores), 
                   'b-', linewidth=2, label='Single-Objective')
    axes[0, 0].plot(range(1, len(multi_scores) + 1), np.maximum.accumulate(multi_scores), 
                   'r-', linewidth=2, label='Multi-Objective')
    axes[0, 0].plot(range(1, len(constrained_scores) + 1), np.maximum.accumulate(constrained_scores), 
                   'g-', linewidth=2, label='Constrained')
    
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('Best Score So Far')
    axes[0, 0].set_title('Optimization Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Parameter sensitivity
    param_names = list(sensitivity_results.keys())
    sensitivities = [sensitivity_results[name]['sensitivity'] for name in param_names]
    
    bars = axes[0, 1].bar(range(len(param_names)), sensitivities, 
                         color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Parameter')
    axes[0, 1].set_ylabel('Sensitivity (Std Dev)')
    axes[0, 1].set_title('Parameter Sensitivity Analysis')
    axes[0, 1].set_xticks(range(len(param_names)))
    axes[0, 1].set_xticklabels([name.split('__')[-1] for name in param_names], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, sens in zip(bars, sensitivities):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{sens:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Parameter value distributions
    all_params = {}
    for trial in single_results.trials:
        for param, value in trial.params.items():
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(value)
    
    # Select top 4 most important parameters
    important_params = sorted(param_names, key=lambda x: sensitivity_results[x]['sensitivity'], reverse=True)[:4]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    for i, param in enumerate(important_params):
        if param in all_params:
            axes[0, 2].hist(all_params[param], bins=10, alpha=0.7, 
                           label=param.split('__')[-1], color=colors[i % len(colors)])
    
    axes[0, 2].set_xlabel('Parameter Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Explored Parameter Distributions')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Score distribution comparison
    all_scores = [single_scores, multi_scores, constrained_scores]
    labels = ['Single-Objective', 'Multi-Objective', 'Constrained']
    colors = ['blue', 'red', 'green']
    
    for scores, label, color in zip(all_scores, labels, colors):
        axes[1, 0].hist(scores, bins=15, alpha=0.6, label=label, color=color)
    
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Parameter correlation heatmap
    if len(all_params) >= 3:
        param_matrix = []
        common_params = list(all_params.keys())[:4]  # Take first 4 parameters
        
        for param in common_params:
            param_matrix.append(all_params[param])
        
        param_matrix = np.array(param_matrix)
        correlation_matrix = np.corrcoef(param_matrix)
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(common_params)))
        axes[1, 1].set_yticks(range(len(common_params)))
        axes[1, 1].set_xticklabels([p.split('__')[-1] for p in common_params], rotation=45)
        axes[1, 1].set_yticklabels([p.split('__')[-1] for p in common_params])
        axes[1, 1].set_title('Parameter Correlation Matrix')
        
        # Add correlation values
        for i in range(len(common_params)):
            for j in range(len(common_params)):
                text = axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                                     ha='center', va='center', 
                                     color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=axes[1, 1])
    
    # 6. Optimization efficiency comparison
    methods = ['Single-Obj', 'Multi-Obj', 'Constrained']
    best_scores = [max(single_scores), max(multi_scores), max(constrained_scores)]
    trial_counts = [len(single_scores), len(multi_scores), len(constrained_scores)]
    
    efficiency = [score/trials for score, trials in zip(best_scores, trial_counts)]
    
    bars = axes[1, 2].bar(methods, efficiency, color=['blue', 'red', 'green'], alpha=0.7)
    axes[1, 2].set_ylabel('Score / Trial (Efficiency)')
    axes[1, 2].set_title('Optimization Efficiency')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                       f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/bayesian_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved Bayesian optimization analysis to ./output/bayesian_optimization_analysis.png")

def main():
    """Run the complete Bayesian optimization cookbook"""
    print("🦙 RLlama Bayesian Optimization Cookbook")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # 1. Single-objective optimization
    single_results, single_time = run_single_objective_optimization()
    
    # 2. Multi-objective optimization
    multi_results = run_multi_objective_optimization()
    
    # 3. Parameter sensitivity analysis
    sensitivity_results = parameter_sensitivity_analysis()
    
    # 4. Constrained optimization
    constrained_results = run_constrained_optimization()
    
    # 5. Create visualizations
    create_optimization_visualizations(single_results, multi_results, sensitivity_results, constrained_results)
    
    # Print final summary
    print("\n" + "="*50)
    print("�� Bayesian Optimization Cookbook Complete!")
    print("="*50)
    
    print(f"\n📊 Optimization Results Summary:")
    print(f"  Single-Objective:")
    print(f"    • Best score: {single_results.best_value:.6f}")
    print(f"    • Optimization time: {single_time:.2f}s")
    print(f"    • Trials: {len(single_results.trials)}")
    
    print(f"\n  Multi-Objective:")
    print(f"    • Best score: {multi_results.best_value:.6f}")
    print(f"    • Trials: {len(multi_results.trials)}")
    
    print(f"\n  Constrained:")
    print(f"    • Best score: {constrained_results.best_value:.6f}")
    print(f"    • Trials: {len(constrained_results.trials)}")
    
    print(f"\n  Parameter Sensitivity:")
    sorted_sensitivity = sorted(sensitivity_results.items(), 
                               key=lambda x: x[1]['sensitivity'], reverse=True)
    for param, result in sorted_sensitivity[:3]:
        print(f"    • {param}: {result['sensitivity']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  • ./output/bayesian_optimization_analysis.png")
    print(f"  • ./output/optimized_single_objective_config.yaml")
    
    print(f"\n🔗 Key Insights:")
    print(f"  • Bayesian optimization finds optimal parameters efficiently")
    print(f"  • Parameter sensitivity varies significantly across components")
    print(f"  • Multi-objective optimization balances competing goals")
    print(f"  • Constraints can be incorporated naturally")

if __name__ == "__main__":
    main()
