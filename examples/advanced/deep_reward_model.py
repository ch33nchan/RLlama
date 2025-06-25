#!/usr/bin/env python3
"""
Example demonstrating neural network reward models in RLlama.
Shows training of MLP and ensemble models with proper evaluation and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
from pathlib import Path
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama.models import MLPRewardModel, EnsembleRewardModel, RewardModelTrainer

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def generate_synthetic_data(n_samples=1000, noise_level=0.1, random_seed=42):
    """
    Generate synthetic data for reward modeling with complex patterns.
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Standard deviation of noise to add
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (states, noisy_rewards, true_rewards)
    """
    np.random.seed(random_seed)
    
    # Generate random states with different distributions
    states = np.random.uniform(-2, 2, (n_samples, 4))
    
    # Complex true reward function with multiple patterns
    def true_reward_function(state):
        x1, x2, x3, x4 = state
        
        # Quadratic terms
        quad_term = -0.3 * x1**2 - 0.2 * x2**2
        
        # Nonlinear interactions
        nonlinear = np.sin(2 * x1) * np.cos(x2) + 0.5 * np.tanh(x3)
        
        # Cross-interactions
        interaction = -0.15 * x1 * x3 + 0.1 * x2 * x4
        
        # Linear terms
        linear = 0.4 * x1 - 0.3 * x2 + 0.2 * x3 - 0.1 * x4
        
        # Combine all terms
        reward = quad_term + nonlinear + interaction + linear + 1.0
        
        return reward
    
    # Calculate true rewards
    true_rewards = np.array([true_reward_function(state) for state in states])
    
    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    noisy_rewards = true_rewards + noise
    
    # Reshape for model compatibility
    return states, noisy_rewards.reshape(-1, 1), true_rewards.reshape(-1, 1)

def create_data_loaders(states, rewards, train_ratio=0.8, batch_size=32, val_batch_size=64):
    """
    Create PyTorch data loaders for training and validation.
    
    Args:
        states: Input states array
        rewards: Target rewards array
        train_ratio: Fraction of data to use for training
        batch_size: Training batch size
        val_batch_size: Validation batch size
        
    Returns:
        Tuple of (train_loader, val_loader, train_data, val_data)
    """
    n_samples = len(states)
    train_size = int(train_ratio * n_samples)
    
    # Split data
    train_states = states[:train_size]
    train_rewards = rewards[:train_size]
    val_states = states[train_size:]
    val_rewards = rewards[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_rewards)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_states),
        torch.FloatTensor(val_rewards)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    
    return train_loader, val_loader, (train_states, train_rewards), (val_states, val_rewards)

def evaluate_model_performance(model, val_states, val_rewards, true_rewards, device):
    """
    Comprehensive evaluation of model performance.
    
    Args:
        model: Trained model to evaluate
        val_states: Validation states
        val_rewards: Noisy validation rewards
        true_rewards: True validation rewards
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    val_states_tensor = torch.FloatTensor(val_states).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'forward') and 'return_uncertainty' in model.forward.__code__.co_varnames:
            # Ensemble model with uncertainty
            predictions, uncertainty = model(val_states_tensor, return_uncertainty=True)
            predictions = predictions.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
        else:
            # Regular model
            predictions = model(val_states_tensor).cpu().numpy()
            uncertainty = None
    
    # Calculate metrics against true rewards
    mse_true = mean_squared_error(true_rewards, predictions)
    r2_true = r2_score(true_rewards, predictions)
    mae_true = np.mean(np.abs(true_rewards - predictions))
    
    # Calculate metrics against noisy rewards
    mse_noisy = mean_squared_error(val_rewards, predictions)
    r2_noisy = r2_score(val_rewards, predictions)
    mae_noisy = np.mean(np.abs(val_rewards - predictions))
    
    # Correlation with true rewards
    correlation = np.corrcoef(true_rewards.flatten(), predictions.flatten())[0, 1]
    
    metrics = {
        'mse_vs_true': mse_true,
        'r2_vs_true': r2_true,
        'mae_vs_true': mae_true,
        'mse_vs_noisy': mse_noisy,
        'r2_vs_noisy': r2_noisy,
        'mae_vs_noisy': mae_noisy,
        'correlation': correlation,
        'predictions': predictions,
        'uncertainty': uncertainty
    }
    
    return metrics

def plot_training_results(mlp_history, ensemble_history, output_dir):
    """
    Plot comprehensive training results.
    
    Args:
        mlp_history: Training history for MLP model
        ensemble_history: Training history for ensemble model
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(mlp_history['train_loss'], label='MLP Train', linewidth=2)
    axes[0, 0].plot(mlp_history['val_loss'], label='MLP Val', linewidth=2)
    axes[0, 0].plot(ensemble_history['train_loss'], label='Ensemble Train', linewidth=2)
    axes[0, 0].plot(ensemble_history['val_loss'], label='Ensemble Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Validation correlation
    if 'val_correlation' in mlp_history:
        axes[0, 1].plot(mlp_history['val_correlation'], label='MLP', linewidth=2)
        axes[0, 1].plot(ensemble_history['val_correlation'], label='Ensemble', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Correlation with True Rewards')
        axes[0, 1].set_title('Validation Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Loss comparison
    final_mlp_loss = mlp_history['val_loss'][-1]
    final_ensemble_loss = ensemble_history['val_loss'][-1]
    
    axes[1, 0].bar(['MLP', 'Ensemble'], [final_mlp_loss, final_ensemble_loss], 
                   color=['skyblue', 'lightcoral'])
    axes[1, 0].set_ylabel('Final Validation Loss')
    axes[1, 0].set_title('Final Model Performance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training time or epochs
    axes[1, 1].bar(['MLP', 'Ensemble'], 
                   [len(mlp_history['train_loss']), len(ensemble_history['train_loss'])],
                   color=['skyblue', 'lightcoral'])
    axes[1, 1].set_ylabel('Training Epochs')
    axes[1, 1].set_title('Training Duration')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_analysis(mlp_metrics, ensemble_metrics, val_data, output_dir):
    """
    Plot detailed prediction analysis.
    
    Args:
        mlp_metrics: MLP evaluation metrics
        ensemble_metrics: Ensemble evaluation metrics
        val_data: Validation data tuple (states, rewards)
        output_dir: Directory to save plots
    """
    val_states, val_rewards = val_data
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sample subset for visualization
    n_viz = min(200, len(val_states))
    viz_indices = np.random.choice(len(val_states), n_viz, replace=False)
    
    # MLP predictions vs true
    axes[0, 0].scatter(val_rewards[viz_indices], mlp_metrics['predictions'][viz_indices], 
                       alpha=0.6, s=30)
    axes[0, 0].plot([val_rewards.min(), val_rewards.max()], 
                    [val_rewards.min(), val_rewards.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('True Rewards')
    axes[0, 0].set_ylabel('MLP Predictions')
    axes[0, 0].set_title(f'MLP: R² = {mlp_metrics["r2_vs_noisy"]:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Ensemble predictions vs true
    axes[0, 1].scatter(val_rewards[viz_indices], ensemble_metrics['predictions'][viz_indices], 
                       alpha=0.6, s=30)
    axes[0, 1].plot([val_rewards.min(), val_rewards.max()], 
                    [val_rewards.min(), val_rewards.max()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Rewards')
    axes[0, 1].set_ylabel('Ensemble Predictions')
    axes[0, 1].set_title(f'Ensemble: R² = {ensemble_metrics["r2_vs_noisy"]:.3f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Prediction errors
    mlp_errors = np.abs(val_rewards.flatten() - mlp_metrics['predictions'].flatten())
    ensemble_errors = np.abs(val_rewards.flatten() - ensemble_metrics['predictions'].flatten())
    
    axes[0, 2].hist(mlp_errors, bins=30, alpha=0.7, label='MLP', density=True)
    axes[0, 2].hist(ensemble_errors, bins=30, alpha=0.7, label='Ensemble', density=True)
    axes[0, 2].set_xlabel('Absolute Error')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Residual analysis
    mlp_residuals = val_rewards.flatten() - mlp_metrics['predictions'].flatten()
    ensemble_residuals = val_rewards.flatten() - ensemble_metrics['predictions'].flatten()
    
    axes[1, 0].scatter(mlp_metrics['predictions'][viz_indices], mlp_residuals[viz_indices], 
                       alpha=0.6, s=30)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 0].set_xlabel('MLP Predictions')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('MLP Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(ensemble_metrics['predictions'][viz_indices], ensemble_residuals[viz_indices], 
                       alpha=0.6, s=30)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[1, 1].set_xlabel('Ensemble Predictions')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Ensemble Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Uncertainty analysis (if available)
    if ensemble_metrics['uncertainty'] is not None:
        uncertainty = ensemble_metrics['uncertainty'].flatten()
        sorted_indices = np.argsort(uncertainty)
        
        axes[1, 2].errorbar(range(len(sorted_indices[::10])), 
                           ensemble_metrics['predictions'].flatten()[sorted_indices[::10]],
                           yerr=uncertainty[sorted_indices[::10]], 
                           fmt='o', alpha=0.6, capsize=3)
        axes[1, 2].scatter(range(len(sorted_indices[::10])), 
                          val_rewards.flatten()[sorted_indices[::10]], 
                          alpha=0.8, color='red', s=20, label='True')
        axes[1, 2].set_xlabel('Sample Index (sorted by uncertainty)')
        axes[1, 2].set_ylabel('Reward Value')
        axes[1, 2].set_title('Ensemble Uncertainty')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Uncertainty\nAvailable', 
                        transform=axes[1, 2].transAxes, ha='center', va='center')
        axes[1, 2].set_title('Uncertainty Analysis')
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Run comprehensive deep reward model example."""
    print("🦙 RLlama Deep Reward Model Example")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic reward data...")
    n_samples = 2000
    states, noisy_rewards, true_rewards = generate_synthetic_data(
        n_samples=n_samples, noise_level=0.15, random_seed=42)
    
    print(f"✓ Generated {n_samples} samples with state dimension {states.shape[1]}")
    print(f"  - Reward range: [{true_rewards.min():.3f}, {true_rewards.max():.3f}]")
    print(f"  - Noise level: 0.15 (SNR: {np.var(true_rewards)/0.15**2:.2f})")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader, val_loader, train_data, val_data = create_data_loaders(
        states, noisy_rewards, train_ratio=0.8, batch_size=32, val_batch_size=64)
    
    train_states, train_rewards = train_data
    val_states, val_rewards = val_data
    val_true_rewards = true_rewards[len(train_states):]
    
    print(f"✓ Training samples: {len(train_states)}")
    print(f"✓ Validation samples: {len(val_states)}")
    
    # Train MLP model
    print("\n🧠 Training MLP Reward Model...")
    mlp_model = MLPRewardModel(
        input_dim=states.shape[1],
        hidden_dims=[128, 64, 32],
        activation=nn.ReLU
    )
    
    mlp_trainer = RewardModelTrainer(
        model=mlp_model,
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    mlp_history = mlp_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=10,
        verbose=True,
        save_best=str(output_dir / "best_mlp_model.pt")
    )
    
    # Train ensemble model
    print("\n🎯 Training Ensemble Reward Model...")
    ensemble_model = EnsembleRewardModel(
        input_dim=states.shape[1],
        hidden_dims=[128, 64, 32],
        num_models=5,
        activation=nn.ReLU
    )
    
    ensemble_trainer = RewardModelTrainer(
        model=ensemble_model,
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    ensemble_history = ensemble_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=10,
        verbose=True,
        save_best=str(output_dir / "best_ensemble_model.pt")
    )
    
    # Load best models for evaluation
    print("\n📈 Evaluating trained models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_mlp = MLPRewardModel.load(str(output_dir / "best_mlp_model.pt"))
    best_ensemble = EnsembleRewardModel.load(str(output_dir / "best_ensemble_model.pt"))
    
    best_mlp = best_mlp.to(device)
    best_ensemble = best_ensemble.to(device)
    
    # Comprehensive evaluation
    mlp_metrics = evaluate_model_performance(
        best_mlp, val_states, val_rewards, val_true_rewards, device)
    ensemble_metrics = evaluate_model_performance(
        best_ensemble, val_states, val_rewards, val_true_rewards, device)
    
    # Print results
    print("\n📊 Model Performance Summary:")
    print("-" * 50)
    print(f"MLP Model:")
    print(f"  - MSE vs True Rewards: {mlp_metrics['mse_vs_true']:.6f}")
    print(f"  - R² vs True Rewards: {mlp_metrics['r2_vs_true']:.4f}")
    print(f"  - Correlation: {mlp_metrics['correlation']:.4f}")
    print(f"  - MAE: {mlp_metrics['mae_vs_true']:.4f}")
    
    print(f"\nEnsemble Model:")
    print(f"  - MSE vs True Rewards: {ensemble_metrics['mse_vs_true']:.6f}")
    print(f"  - R² vs True Rewards: {ensemble_metrics['r2_vs_true']:.4f}")
    print(f"  - Correlation: {ensemble_metrics['correlation']:.4f}")
    print(f"  - MAE: {ensemble_metrics['mae_vs_true']:.4f}")
    
    if ensemble_metrics['uncertainty'] is not None:
        mean_uncertainty = np.mean(ensemble_metrics['uncertainty'])
        print(f"  - Mean Uncertainty: {mean_uncertainty:.4f}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_training_results(mlp_history, ensemble_history, output_dir)
    plot_prediction_analysis(mlp_metrics, ensemble_metrics, val_data, output_dir)
    
    # Save detailed results
    results_summary = {
        'mlp_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in mlp_metrics.items() if k not in ['predictions', 'uncertainty']},
        'ensemble_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in ensemble_metrics.items() if k not in ['predictions', 'uncertainty']},
        'training_info': {
            'n_samples': n_samples,
            'train_samples': len(train_states),
            'val_samples': len(val_states),
            'mlp_epochs': len(mlp_history['train_loss']),
            'ensemble_epochs': len(ensemble_history['train_loss']),
            'device': str(device)
        }
    }
    
    import json
    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Deep reward model example completed successfully!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print(f"📊 Visualizations: training_curves.png, prediction_analysis.png")
    print(f"💾 Models saved: best_mlp_model.pt, best_ensemble_model.pt")
    print(f"📋 Summary: results_summary.json")

if __name__ == "__main__":
    main()
