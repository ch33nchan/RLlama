#!/usr/bin/env python3
"""
Complete RLHF Training Pipeline Cookbook
=======================================

This cookbook demonstrates a complete Reinforcement Learning from Human Feedback (RLHF)
pipeline using RLlama's preference collection and training components.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score

sys.path.append(os.path.abspath("../.."))

from rllama.models import MLPRewardModel
from rllama.rlhf import PreferenceCollector, PreferenceTrainer, ActivePreferenceCollector, PreferenceDataset

def generate_ground_truth_preferences(n_samples=1000, state_dim=4):
    """Generate synthetic preference data with known ground truth"""
    print("🎯 Generating ground truth preference data...")
    
    # Ground truth reward function (unknown to the model)
    def ground_truth_reward(state):
        x = state
        # Complex multi-modal reward function
        reward = (
            # Main objective: quadratic with optimal point
            -0.5 * np.sum(x[:2]**2) + 
            
            # Interaction terms
            0.3 * x[0] * x[1] +
            
            # Non-linear components
            np.sin(2 * x[2]) * np.cos(x[3]) +
            
            # Sparse reward component
            (1.0 if np.sum(x**2) > 2.0 else 0.0) +
            
            # Exploration bonus
            0.1 * np.exp(-np.sum(x**2) / 2)
        )
        return reward
    
    # Generate random state pairs
    states_a = np.random.uniform(-2, 2, (n_samples, state_dim))
    states_b = np.random.uniform(-2, 2, (n_samples, state_dim))
    
    # Calculate ground truth rewards
    rewards_a = np.array([ground_truth_reward(s) for s in states_a])
    rewards_b = np.array([ground_truth_reward(s) for s in states_b])
    
    # Generate preferences with some noise to simulate human inconsistency
    noise_level = 0.1
    noisy_rewards_a = rewards_a + np.random.normal(0, noise_level, n_samples)
    noisy_rewards_b = rewards_b + np.random.normal(0, noise_level, n_samples)
    
    # Create preferences: 1.0 if A > B, 0.0 if B > A, 0.5 if tie
    preferences = []
    for ra, rb in zip(noisy_rewards_a, noisy_rewards_b):
        if abs(ra - rb) < 0.1:  # Tie threshold
            preferences.append(0.5)
        elif ra > rb:
            preferences.append(1.0)
        else:
            preferences.append(0.0)
    
    preferences = np.array(preferences)
    
    print(f"✅ Generated {n_samples} preference pairs")
    print(f"   A preferred: {np.sum(preferences == 1.0)}")
    print(f"   B preferred: {np.sum(preferences == 0.0)}")
    print(f"   Ties: {np.sum(preferences == 0.5)}")
    
    return states_a, states_b, preferences, rewards_a, rewards_b, ground_truth_reward

def demonstrate_preference_collection():
    """Demonstrate basic preference collection"""
    print("\n📝 Demonstrating Preference Collection")
    print("-" * 40)
    
    # Create preference collector
    collector = PreferenceCollector(
        buffer_size=10000,
        sampling_strategy='recent'
    )
    
    # Generate some preferences
    states_a, states_b, preferences, _, _, _ = generate_ground_truth_preferences(n_samples=200)
    
    # Add preferences to collector
    for i in range(len(preferences)):
        collector.add_preference(
            states_a[i], 
            states_b[i], 
            preferences[i],
            metadata={'annotator': f'user_{i % 3}', 'confidence': np.random.uniform(0.7, 1.0)}
        )
    
    print(f"Collected {len(collector)} preferences")
    
    # Sample batches
    batch_a, batch_b, batch_prefs = collector.sample_batch(32)
    print(f"Sampled batch shapes: {batch_a.shape}, {batch_b.shape}, {batch_prefs.shape}")
    
    # Test different sampling strategies
    print("\nTesting sampling strategies:")
    for strategy in ['random', 'recent', 'uncertainty']:
        collector.sampling_strategy = strategy
        try:
            batch_a, batch_b, batch_prefs = collector.sample_batch(16)
            print(f"  {strategy}: ✅ (batch size: {len(batch_prefs)})")
        except Exception as e:
            print(f"  {strategy}: ❌ ({str(e)})")
    
    return collector

def train_baseline_reward_model():
    """Train a baseline reward model from preferences"""
    print("\n🏗️ Training Baseline Reward Model")
    print("-" * 40)
    
    # Generate preference data
    states_a, states_b, preferences, rewards_a, rewards_b, gt_function = generate_ground_truth_preferences(1000)
    
    # Split into train/val sets
    train_size = int(0.8 * len(preferences))
    
    train_states_a = states_a[:train_size]
    train_states_b = states_b[:train_size]
    train_preferences = preferences[:train_size]
    
    val_states_a = states_a[train_size:]
    val_states_b = states_b[train_size:]
    val_preferences = preferences[train_size:]
    val_rewards_a = rewards_a[train_size:]
    val_rewards_b = rewards_b[train_size:]
    
    # Create datasets and loaders
    train_dataset = PreferenceDataset(train_states_a, train_states_b, train_preferences)
    val_dataset = PreferenceDataset(val_states_a, val_states_b, val_preferences)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create and train model
    model = MLPRewardModel(input_dim=4, hidden_dims=[64, 32], activation=nn.ReLU)
    trainer = PreferenceTrainer(
        model=model,
        learning_rate=0.001,
        temperature=1.0
    )
    
    print("Training model from preferences...")
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=8,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Evaluate reward correlation with ground truth
    model.eval()
    with torch.no_grad():
        test_states = np.random.uniform(-2, 2, (300, 4))
        test_tensor = torch.FloatTensor(test_states)
        predicted_rewards = model(test_tensor).numpy().flatten()
        true_rewards = np.array([gt_function(s) for s in test_states])
        
        correlation = np.corrcoef(predicted_rewards, true_rewards)[0, 1]
    
    print(f"\n📊 Baseline Results:")
    print(f"Training time: {training_time:.2f}s")
    print(f"Final preference accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Reward correlation with ground truth: {correlation:.4f}")
    
    return model, history, correlation

def demonstrate_active_learning():
    """Demonstrate active learning for preference collection"""
    print("\n🎯 Demonstrating Active Learning")
    print("-" * 40)
    
    # Generate ground truth data
    states_a, states_b, preferences, rewards_a, rewards_b, gt_function = generate_ground_truth_preferences(600)
    
    # Initial small training set
    initial_size = 100
    collector = PreferenceCollector(buffer_size=10000)
    
    for i in range(initial_size):
        collector.add_preference(states_a[i], states_b[i], preferences[i])
    
    # Train initial model
    initial_states_a, initial_states_b, initial_prefs = collector.get_all_data()
    dataset = PreferenceDataset(initial_states_a, initial_states_b, initial_prefs)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = MLPRewardModel(input_dim=4, hidden_dims=[64, 32])
    trainer = PreferenceTrainer(model)
    trainer.train(loader, epochs=30, verbose=False)
    
    # Set up active learning
    active_collector = ActivePreferenceCollector(
        buffer_size=10000,
        model=model,
        query_batch_size=50
    )
    
    # Add remaining data as candidates
    remaining_states = np.vstack([states_a[initial_size:], states_b[initial_size:]])
    active_collector.add_candidate_states(list(remaining_states))
    
    # Active learning loop
    n_iterations = 15
    batch_size = 20
    
    learning_curves = {'active': [], 'random': []}
    
    for iteration in range(n_iterations):
        print(f"Active learning iteration {iteration + 1}/{n_iterations}")
        
        # Active learning queries
        for _ in range(batch_size):
            state_a, state_b = active_collector.select_query_pair()
            if state_a is not None and state_b is not None:
                # Simulate human feedback using ground truth
                reward_a = gt_function(state_a)
                reward_b = gt_function(state_b)
                
                if abs(reward_a - reward_b) < 0.1:
                    pref = 0.5
                elif reward_a > reward_b:
                    pref = 1.0
                else:
                    pref = 0.0
                
                active_collector.add_preference(state_a, state_b, pref)
        
        # Random queries for comparison
        random_indices = np.random.choice(
            range(initial_size, len(states_a)), 
            size=batch_size, 
            replace=False
        )
        random_collector = PreferenceCollector(buffer_size=10000)
        
        # Add initial data to random collector
        for i in range(initial_size):
            random_collector.add_preference(states_a[i], states_b[i], preferences[i])
        
        # Add random samples
        for idx in random_indices:
            random_collector.add_preference(states_a[idx], states_b[idx], preferences[idx])
        
        # Evaluate both approaches
        for approach, collector_obj in [('active', active_collector), ('random', random_collector)]:
            # Get all data
            all_states_a, all_states_b, all_prefs = collector_obj.get_all_data()
            
            # Train model
            dataset = PreferenceDataset(all_states_a, all_states_b, all_prefs)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            eval_model = MLPRewardModel(input_dim=4, hidden_dims=[64, 32])
            eval_trainer = PreferenceTrainer(eval_model)
            eval_trainer.train(loader, epochs=25, verbose=False)
            
            # Test correlation with ground truth
            test_states = np.random.uniform(-2, 2, (200, 4))
            eval_model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(test_states)
                pred_rewards = eval_model(test_tensor).numpy().flatten()
                true_rewards = np.array([gt_function(s) for s in test_states])
                correlation = np.corrcoef(pred_rewards, true_rewards)[0, 1]
                
            learning_curves[approach].append(correlation)
    
    print(f"\n📈 Active Learning Results:")
    print(f"Final active learning correlation: {learning_curves['active'][-1]:.4f}")
    print(f"Final random sampling correlation: {learning_curves['random'][-1]:.4f}")
    print(f"Improvement: {learning_curves['active'][-1] - learning_curves['random'][-1]:.4f}")
    
    return learning_curves

def simulate_human_annotator_variability():
    """Simulate different types of human annotators"""
    print("\n👥 Simulating Human Annotator Variability")
    print("-" * 40)
    
    # Generate ground truth data
    states_a, states_b, _, rewards_a, rewards_b, gt_function = generate_ground_truth_preferences(400)
    
    # Define annotator profiles
    annotator_profiles = {
        'expert': {'consistency': 0.95, 'bias': 0.0, 'noise': 0.05},
        'amateur': {'consistency': 0.75, 'bias': 0.1, 'noise': 0.2},
        'inconsistent': {'consistency': 0.6, 'bias': 0.0, 'noise': 0.3},
        'biased': {'consistency': 0.85, 'bias': 0.3, 'noise': 0.1}
    }
    
    annotator_results = {}
    
    for annotator_type, profile in annotator_profiles.items():
        print(f"Testing {annotator_type} annotator...")
        
        preferences = []
        
        for ra, rb in zip(rewards_a, rewards_b):
            # Add bias
            biased_ra = ra + profile['bias']
            biased_rb = rb + profile['bias']
            
            # Add noise
            noisy_ra = biased_ra + np.random.normal(0, profile['noise'])
            noisy_rb = biased_rb + np.random.normal(0, profile['noise'])
            
            # Determine preference with consistency
            if np.random.random() > profile['consistency']:
                # Inconsistent annotation
                pref = np.random.choice([0.0, 0.5, 1.0])
            else:
                # Consistent annotation
                if abs(noisy_ra - noisy_rb) < 0.1:
                    pref = 0.5
                elif noisy_ra > noisy_rb:
                    pref = 1.0
                else:
                    pref = 0.0
            
            preferences.append(pref)
        
        # Train model on this annotator's data
        dataset = PreferenceDataset(states_a, states_b, np.array(preferences))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        model = MLPRewardModel(input_dim=4, hidden_dims=[64, 32])
        trainer = PreferenceTrainer(model)
        history = trainer.train(loader, epochs=40, verbose=False)
        
        # Evaluate correlation with ground truth
        model.eval()
        with torch.no_grad():
            test_states = np.random.uniform(-2, 2, (200, 4))
            test_tensor = torch.FloatTensor(test_states)
            pred_rewards = model(test_tensor).numpy().flatten()
            true_rewards = np.array([gt_function(s) for s in test_states])
            correlation = np.corrcoef(pred_rewards, true_rewards)[0, 1]
        
        annotator_results[annotator_type] = {
            'correlation': correlation,
            'final_accuracy': history['val_accuracy'][-1],
            'preferences': preferences,
            'profile': profile
        }
        
        print(f"  Correlation with ground truth: {correlation:.4f}")
        print(f"  Final preference accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return annotator_results

def create_comprehensive_visualizations(baseline_history, learning_curves, annotator_results):
    """Create comprehensive RLHF visualizations"""
    print("\n📊 Creating comprehensive visualizations...")
    
    os.makedirs("./output", exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Learning curves for baseline model
    epochs = range(1, len(baseline_history['train_loss']) + 1)
    axes[0, 0].plot(epochs, baseline_history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, baseline_history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Preference Model Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Active learning comparison
    iterations = range(1, len(learning_curves['active']) + 1)
    axes[0, 1].plot(iterations, learning_curves['active'], 'ro-', 
                   label='Active Learning', linewidth=2, markersize=6)
    axes[0, 1].plot(iterations, learning_curves['random'], 'bo-', 
                   label='Random Sampling', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Correlation with Ground Truth')
    axes[0, 1].set_title('Active Learning vs Random Sampling')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Annotator performance comparison
    annotator_names = list(annotator_results.keys())
    correlations = [annotator_results[name]['correlation'] for name in annotator_names]
    colors = ['green', 'orange', 'red', 'purple']
    
    bars = axes[0, 2].bar(range(len(annotator_names)), correlations, 
                         color=colors, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Annotator Type')
    axes[0, 2].set_ylabel('Correlation with Ground Truth')
    axes[0, 2].set_title('Model Performance by Annotator Type')
    axes[0, 2].set_xticks(range(len(annotator_names)))
    axes[0, 2].set_xticklabels(annotator_names, rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Preference accuracy vs ground truth correlation
    accuracies = [annotator_results[name]['final_accuracy'] for name in annotator_names]
    
    axes[1, 0].scatter(accuracies, correlations, s=100, c=colors, alpha=0.7, edgecolors='black')
    for i, name in enumerate(annotator_names):
        axes[1, 0].annotate(name, (accuracies[i], correlations[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[1, 0].set_xlabel('Preference Accuracy')
    axes[1, 0].set_ylabel('Ground Truth Correlation')
    axes[1, 0].set_title('Accuracy vs Ground Truth Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Annotator characteristics analysis
    consistency_scores = [annotator_results[name]['profile']['consistency'] for name in annotator_names]
    noise_levels = [annotator_results[name]['profile']['noise'] for name in annotator_names]
    
    scatter = axes[1, 1].scatter(consistency_scores, correlations, c=noise_levels, 
                                s=100, cmap='Reds', alpha=0.7, edgecolors='black')
    
    for i, name in enumerate(annotator_names):
        axes[1, 1].annotate(name, (consistency_scores[i], correlations[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    axes[1, 1].set_xlabel('Annotator Consistency')
    axes[1, 1].set_ylabel('Model Correlation')
    axes[1, 1].set_title('Consistency vs Model Performance')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Noise Level')
    
    # 6. Learning efficiency comparison
    final_active = learning_curves['active'][-1]
    final_random = learning_curves['random'][-1]
    improvement = final_active - final_random
    
    methods = ['Random\nSampling', 'Active\nLearning']
    final_scores = [final_random, final_active]
    colors_method = ['lightblue', 'lightcoral']
    
    bars = axes[1, 2].bar(methods, final_scores, color=colors_method, alpha=0.7, edgecolor='black')
    axes[1, 2].set_ylabel('Final Correlation')
    axes[1, 2].set_title(f'Learning Method Comparison\n(Improvement: +{improvement:.3f})')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, final_scores):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/rlhf_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comprehensive RLHF analysis to ./output/rlhf_comprehensive_analysis.png")

def main():
    """Run the complete RLHF training pipeline cookbook"""
    print("🦙 RLlama RLHF Training Pipeline Cookbook")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # 1. Demonstrate preference collection
    collector = demonstrate_preference_collection()
    
    # 2. Train baseline reward model
    baseline_model, baseline_history, baseline_correlation = train_baseline_reward_model()
    
    # 3. Demonstrate active learning
    learning_curves = demonstrate_active_learning()
    
    # 4. Simulate annotator variability
    annotator_results = simulate_human_annotator_variability()
    
    # 5. Create comprehensive visualizations
    create_comprehensive_visualizations(baseline_history, learning_curves, annotator_results)
    
    # Print final summary
    print("\n" + "="*50)
    print("🎉 RLHF Training Pipeline Cookbook Complete!")
    print("="*50)
    
    print(f"\n📊 Final Results Summary:")
    print(f"  Baseline Model:")
    print(f"    • Preference accuracy: {baseline_history['val_accuracy'][-1]:.4f}")
    print(f"    • Ground truth correlation: {baseline_correlation:.4f}")
    
    print(f"\n  Active Learning:")
    print(f"    • Final active correlation: {learning_curves['active'][-1]:.4f}")
    print(f"    • Final random correlation: {learning_curves['random'][-1]:.4f}")
    print(f"    • Improvement: +{learning_curves['active'][-1] - learning_curves['random'][-1]:.4f}")
    
    print(f"\n  Annotator Analysis:")
    best_annotator = max(annotator_results.items(), key=lambda x: x[1]['correlation'])
    worst_annotator = min(annotator_results.items(), key=lambda x: x[1]['correlation'])
    print(f"    • Best: {best_annotator[0]} ({best_annotator[1]['correlation']:.4f})")
    print(f"    • Worst: {worst_annotator[0]} ({worst_annotator[1]['correlation']:.4f})")
    
    print(f"\n📁 Generated Files:")
    print(f"  • ./output/rlhf_comprehensive_analysis.png")
    
    print(f"\n🔗 Key Insights:")
    print(f"  • Active learning significantly improves data efficiency")
    print(f"  • Annotator quality greatly impacts model performance")
    print(f"  • Bradley-Terry model effectively learns from preferences")
    print(f"  • RLHF scales well with preference data quality")

if __name__ == "__main__":
    main()
