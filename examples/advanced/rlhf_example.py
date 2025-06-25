#!/usr/bin/env python3
"""
Example demonstrating Reinforcement Learning from Human Feedback (RLHF) in RLlama.
Shows preference collection, model training, and active learning with comprehensive evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama.models import MLPRewardModel
from rllama.rlhf import PreferenceDataset, PreferenceTrainer, PreferenceCollector, ActivePreferenceCollector

# Configure logging and plotting
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
plt.style.use('default')
sns.set_palette("husl")

class GroundTruthRewardFunction:
    """Ground truth reward function for generating synthetic preferences."""
    
    def __init__(self, state_dim: int = 4, complexity: str = "medium"):
        """
        Initialize ground truth reward function.
        
        Args:
            state_dim: Dimension of state space
            complexity: Complexity level ("simple", "medium", "complex")
        """
        self.state_dim = state_dim
        self.complexity = complexity
        
        # Set random weights for reproducibility
        np.random.seed(42)
        self.linear_weights = np.random.uniform(-1, 1, state_dim)
        self.quadratic_weights = np.random.uniform(-0.5, 0.5, state_dim)
        self.interaction_matrix = np.random.uniform(-0.3, 0.3, (state_dim, state_dim))
        
        # Make interaction matrix symmetric
        self.interaction_matrix = (self.interaction_matrix + self.interaction_matrix.T) / 2
        
    def __call__(self, state: np.ndarray) -> float:
        """
        Calculate ground truth reward for a state.
        
        Args:
            state: State vector
            
        Returns:
            Ground truth reward value
        """
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Linear terms
        linear_reward = np.dot(state, self.linear_weights)
        
        if self.complexity == "simple":
            return float(linear_reward[0] if hasattr(linear_reward, '__len__') else linear_reward)
        
        # Quadratic terms
        quadratic_reward = np.sum(self.quadratic_weights * state**2, axis=1)
        
        if self.complexity == "medium":
            total = linear_reward + quadratic_reward
            return float(total[0] if hasattr(total, '__len__') else total)
        
        # Complex: add interactions and nonlinearities
        interaction_reward = np.sum(state @ self.interaction_matrix * state, axis=1)
        
        # Nonlinear terms
        nonlinear_reward = (
            0.3 * np.sin(2 * state[:, 0]) +
            0.2 * np.cos(state[:, 1]) +
            0.1 * np.tanh(state[:, 2]) +
            0.15 * np.exp(-0.5 * state[:, 3]**2)
        )
        
        total = linear_reward + quadratic_reward + interaction_reward + nonlinear_reward
        return float(total[0] if hasattr(total, '__len__') else total)

def generate_synthetic_preferences(n_samples: int = 1000, 
                                 state_dim: int = 4, 
                                 noise_level: float = 0.1,
                                 complexity: str = "medium") -> Tuple[np.ndarray, np.ndarray, np.ndarray, GroundTruthRewardFunction]:
    """
    Generate synthetic preference data based on a ground truth reward function.
    
    Args:
        n_samples: Number of preference pairs to generate
        state_dim: Dimension of state space
        noise_level: Standard deviation of noise to add to preferences
        complexity: Complexity of ground truth function
        
    Returns:
        Tuple of (states_a, states_b, preferences, ground_truth_function)
    """
    # Create ground truth reward function
    gt_reward_fn = GroundTruthRewardFunction(state_dim, complexity)
    
    # Generate random states
    states = np.random.uniform(-2, 2, (n_samples * 2, state_dim))
    
    # Calculate true rewards for all states
    true_rewards = np.array([gt_reward_fn(state) for state in states])
    
    # Add noise to rewards for preference generation
    noisy_rewards = true_rewards + np.random.normal(0, noise_level, len(true_rewards))
    
    # Create preference pairs
    states_a = []
    states_b = []
    preferences = []
    preference_margins = []
    
    for i in range(n_samples):
        # Select two random states
        idx_a = i * 2
        idx_b = i * 2 + 1
        
        state_a = states[idx_a]
        state_b = states[idx_b]
        
        reward_a = noisy_rewards[idx_a]
        reward_b = noisy_rewards[idx_b]
        
        # Calculate preference margin
        margin = abs(reward_a - reward_b)
        preference_margins.append(margin)
        
        # Determine preference with some probability based on margin
        # Larger margins lead to more confident preferences
        confidence = 1 / (1 + np.exp(-5 * margin))  # Sigmoid confidence
        
        if np.random.random() < confidence:
            # Clear preference
            if reward_a > reward_b:
                pref = 1.0  # A is preferred
            elif reward_a < reward_b:
                pref = 0.0  # B is preferred
            else:
                pref = 0.5  # Tie (rare)
        else:
            # Random preference (noise)
            pref = float(np.random.choice([0.0, 1.0]))
            
        states_a.append(state_a)
        states_b.append(state_b)
        preferences.append(pref)
    
    return (np.array(states_a), np.array(states_b), 
            np.array(preferences), gt_reward_fn)

def evaluate_learned_reward(model: MLPRewardModel, 
                          gt_reward_fn: GroundTruthRewardFunction,
                          n_test: int = 500,
                          state_dim: int = 4) -> Dict[str, float]:
    """
    Evaluate learned reward model against ground truth.
    
    Args:
        model: Trained reward model
        gt_reward_fn: Ground truth reward function
        n_test: Number of test samples
        state_dim: Dimension of state space
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Generate test states
    test_states = np.random.uniform(-2, 2, (n_test, state_dim))
    
    # Calculate true rewards
    true_rewards = np.array([gt_reward_fn(state) for state in test_states])
    
    # Predict with model
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_states).to(device)
        pred_rewards = model(test_tensor).cpu().numpy().flatten()
    
    # Calculate metrics
    mse = np.mean((true_rewards - pred_rewards) ** 2)
    mae = np.mean(np.abs(true_rewards - pred_rewards))
    
    # Correlation
    correlation = np.corrcoef(true_rewards, pred_rewards)[0, 1]
    
    # R-squared
    ss_res = np.sum((true_rewards - pred_rewards) ** 2)
    ss_tot = np.sum((true_rewards - np.mean(true_rewards)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Ranking correlation (Spearman)
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(true_rewards, pred_rewards)
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'r_squared': r_squared,
        'spearman_correlation': spearman_corr
    }

def plot_comprehensive_results(history: Dict[str, List[float]], 
                             evaluation_metrics: Dict[str, float],
                             true_rewards: np.ndarray,
                             pred_rewards: np.ndarray,
                             output_dir: Path) -> None:
    """
    Create comprehensive visualization of RLHF results.
    
    Args:
        history: Training history
        evaluation_metrics: Model evaluation metrics
        true_rewards: Ground truth rewards for test set
        pred_rewards: Predicted rewards for test set
        output_dir: Directory to save plots
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Training curves
    ax1 = plt.subplot(2, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Accuracy curves
    ax2 = plt.subplot(2, 4, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Preference Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # True vs Predicted scatter
    ax3 = plt.subplot(2, 4, 3)
    plt.scatter(true_rewards, pred_rewards, alpha=0.6, s=30)
    min_val = min(true_rewards.min(), pred_rewards.min())
    max_val = max(true_rewards.max(), pred_rewards.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    plt.xlabel('True Reward')
    plt.ylabel('Predicted Reward')
    plt.title(f'True vs Predicted\n(r={evaluation_metrics["correlation"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    ax4 = plt.subplot(2, 4, 4)
    residuals = true_rewards - pred_rewards
    plt.scatter(pred_rewards, residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Reward')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    ax5 = plt.subplot(2, 4, 5)
    errors = np.abs(residuals)
    plt.hist(errors, bins=30, alpha=0.7, density=True, edgecolor='black')
    plt.xlabel('Absolute Error')
    plt.ylabel('Density')
    plt.title(f'Error Distribution\n(MAE={evaluation_metrics["mae"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Metrics bar plot
    ax6 = plt.subplot(2, 4, 6)
    metrics_names = ['Correlation', 'R²', 'Spearman']
    metrics_values = [
        evaluation_metrics['correlation'],
        evaluation_metrics['r_squared'],
        evaluation_metrics['spearman_correlation']
    ]
    bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.ylabel('Score')
    plt.title('Evaluation Metrics')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Learning curve (final validation accuracy)
    ax7 = plt.subplot(2, 4, 7)
    final_epochs = min(20, len(history['val_accuracy']))
    epochs_subset = list(range(len(history['val_accuracy']) - final_epochs, len(history['val_accuracy'])))
    acc_subset = history['val_accuracy'][-final_epochs:]
    plt.plot(epochs_subset, acc_subset, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Final Training Phase')
    plt.grid(True, alpha=0.3)
    
    # Prediction confidence
    ax8 = plt.subplot(2, 4, 8)
    # Sort by prediction confidence (distance from mean)
    pred_mean = np.mean(pred_rewards)
    confidence = np.abs(pred_rewards - pred_mean)
    sorted_indices = np.argsort(confidence)
    
    n_show = min(100, len(sorted_indices))
    show_indices = sorted_indices[:n_show]
    
    plt.scatter(range(n_show), true_rewards[show_indices], 
               alpha=0.7, label='True', s=30)
    plt.scatter(range(n_show), pred_rewards[show_indices], 
               alpha=0.7, label='Predicted', s=30)
    plt.xlabel('Sample (sorted by prediction confidence)')
    plt.ylabel('Reward Value')
    plt.title('Low Confidence Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rlhf_comprehensive_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run comprehensive RLHF example."""
    print("🦙 RLlama RLHF Example")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configuration
    config = {
        'n_samples': 2000,
        'state_dim': 4,
        'noise_level': 0.15,
        'complexity': 'complex',
        'hidden_dims': [128, 64, 32],
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 80,
        'early_stopping_patience': 10
    }
    
    # Generate synthetic preference data
    print("\n📊 Generating synthetic preference data...")
    states_a, states_b, preferences, gt_reward_fn = generate_synthetic_preferences(
        n_samples=config['n_samples'],
        state_dim=config['state_dim'],
        noise_level=config['noise_level'],
        complexity=config['complexity']
    )
    
    print(f"✓ Generated {len(preferences)} preference pairs")
    print(f"  - State dimension: {config['state_dim']}")
    print(f"  - Noise level: {config['noise_level']}")
    print(f"  - Complexity: {config['complexity']}")
    print(f"  - Preference distribution: {np.mean(preferences):.3f} (0.5 = balanced)")
    
    # Create preference collector
    print("\n🗂️ Setting up preference collection...")
    collector = PreferenceCollector(buffer_size=10000, sampling_strategy='random')
    
    # Add preferences to collector
    for i in range(len(preferences)):
        collector.add_preference(states_a[i], states_b[i], preferences[i])
    
    print(f"✓ Added {len(preferences)} preferences to collector")
    
    # Split into train/validation sets
    all_states_a, all_states_b, all_prefs = collector.get_all_data()
    train_size = int(0.8 * len(all_prefs))
    
    # Create datasets and dataloaders
    train_dataset = PreferenceDataset(
        all_states_a[:train_size], 
        all_states_b[:train_size], 
        all_prefs[:train_size]
    )
    val_dataset = PreferenceDataset(
        all_states_a[train_size:], 
        all_states_b[train_size:], 
        all_prefs[train_size:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create and train reward model from preferences
    print("\n�� Training reward model from preferences...")
    reward_model = MLPRewardModel(
        input_dim=config['state_dim'],
        hidden_dims=config['hidden_dims'],
        activation=nn.ReLU
    )
    
    trainer = PreferenceTrainer(
        model=reward_model,
        learning_rate=config['learning_rate'],
        temperature=0.5
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        verbose=True,
        save_best=str(output_dir / "best_preference_model.pt")
    )
    
    # Load best model for evaluation
    best_model = MLPRewardModel.load(str(output_dir / "best_preference_model.pt"))
    
    # Comprehensive evaluation
    print("\n📈 Evaluating learned reward model...")
    evaluation_metrics = evaluate_learned_reward(
        best_model, gt_reward_fn, n_test=1000, state_dim=config['state_dim']
    )
    
    print("📊 Model Performance:")
    print("-" * 30)
    for metric, value in evaluation_metrics.items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Active learning demonstration
    print("\n🎯 Demonstrating active preference learning...")
    active_collector = ActivePreferenceCollector(
        buffer_size=10000,
        sampling_strategy='uncertainty',
        model=best_model
    )
    
    # Generate candidate states for active learning
    n_candidates = 500
    candidate_states = np.random.uniform(-2, 2, (n_candidates, config['state_dim']))
    active_collector.add_candidate_states(list(candidate_states))
    
    print(f"✓ Generated {n_candidates} candidate states")
    
    # Select most informative pairs
    n_queries = 15
    active_learning_results = []
    
    print(f"🔍 Selecting {n_queries} most informative pairs:")
    for i in range(n_queries):
        state_a, state_b = active_collector.select_query_pair()
        
        if state_a is None or state_b is None:
            print(f"  Query {i+1}: No more candidate pairs available")
            break
        
        # Simulate human feedback using ground truth
        reward_a = gt_reward_fn(state_a)
        reward_b = gt_reward_fn(state_b)
        
        if reward_a > reward_b:
            pref = 1.0
            pref_str = "A"
        elif reward_a < reward_b:
            pref = 0.0
            pref_str = "B"
        else:
            pref = 0.5
            pref_str = "Tie"
        
        # Add preference to collector
        active_collector.add_preference(state_a, state_b, pref)
        
        # Store results
        active_learning_results.append({
            'query': i + 1,
            'reward_a': reward_a,
            'reward_b': reward_b,
            'preference': pref_str,
            'margin': abs(reward_a - reward_b)
        })
        
        print(f"  Query {i+1}: Rewards: A={reward_a:.3f}, B={reward_b:.3f} → {pref_str}")
    
    # Generate test data for visualization
    test_states = np.random.uniform(-2, 2, (200, config['state_dim']))
    true_test_rewards = np.array([gt_reward_fn(state) for state in test_states])
    
    device = next(best_model.parameters()).device
    best_model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_states).to(device)
        pred_test_rewards = best_model(test_tensor).cpu().numpy().flatten()
    
    # Create comprehensive visualizations
    print("\n📊 Generating comprehensive visualizations...")
    plot_comprehensive_results(
        history, evaluation_metrics, true_test_rewards, pred_test_rewards, output_dir
    )
    
    # Save detailed results
    results_summary = {
        'config': config,
        'evaluation_metrics': evaluation_metrics,
        'training_history': {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'epochs_trained': len(history['train_loss'])
        },
        'active_learning_results': active_learning_results
    }
    
    with open(output_dir / "rlhf_results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ RLHF example completed successfully!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print(f"📊 Comprehensive visualization: rlhf_comprehensive_results.png")
    print(f"💾 Best model: best_preference_model.pt")
    print(f"📋 Summary: rlhf_results_summary.json")
    
    # Print final summary
    print(f"\n🎯 Final Performance Summary:")
    print(f"  Correlation with ground truth: {evaluation_metrics['correlation']:.3f}")
    print(f"  Mean Absolute Error: {evaluation_metrics['mae']:.3f}")
    print(f"  R² Score: {evaluation_metrics['r_squared']:.3f}")
    print(f"  Training completed in {len(history['train_loss'])} epochs")

if __name__ == "__main__":
    main()
