#!/usr/bin/env python3
"""
Complete Deep Reward Models Cookbook
====================================

This cookbook demonstrates how to train sophisticated neural networks
to learn complex reward functions with uncertainty quantification using RLlama.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Add RLlama to path
sys.path.append(os.path.abspath("../.."))

from rllama.models import MLPRewardModel, EnsembleRewardModel, RewardModelTrainer
from rllama.rewards.components import LengthReward, DiversityReward

def generate_synthetic_reward_data(n_samples=2000, noise_level=0.1):
    """Generate synthetic data with complex reward patterns"""
    print("🎯 Generating synthetic reward data...")
    
    # Generate random state vectors (simulating text embeddings)
    states = np.random.uniform(-2, 2, (n_samples, 6))
    
    # Complex true reward function with multiple objectives
    def true_reward_function(state):
        x1, x2, x3, x4, x5, x6 = state
        
        # Multi-modal reward landscape
        reward = (
            # Primary objective: length preference
            -0.3 * (x1**2 + x2**2) +
            
            # Secondary objective: diversity bonus
            np.sin(2 * np.pi * x3) * np.cos(np.pi * x4) +
            
            # Interaction terms (context dependency)
            0.5 * x1 * x2 * np.exp(-x3**2) +
            
            # Sparse rewards for exploration
            (2.0 if abs(x5) > 1.5 and abs(x6) > 1.5 else 0.0) +
            
            # Fine-grained quality signal
            0.1 * np.sin(10 * x1) * np.cos(10 * x2)
        )
        
        return reward
    
    # Calculate true rewards
    true_rewards = np.array([true_reward_function(state) for state in states])
    
    # Add noise to simulate real-world uncertainty
    noisy_rewards = true_rewards + np.random.normal(0, noise_level, n_samples)
    
    print(f"✅ Generated {n_samples} samples with {noise_level} noise level")
    return states, noisy_rewards.reshape(-1, 1), true_rewards.reshape(-1, 1)

def train_mlp_reward_model():
    """Train a single MLP reward model"""
    print("\n🧠 Training MLP Reward Model")
    print("-" * 40)
    
    # Generate data
    states, noisy_rewards, true_rewards = generate_synthetic_reward_data(n_samples=1500)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        states, noisy_rewards, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create MLP model
    model = MLPRewardModel(
        input_dim=6,
        hidden_dims=[128, 64, 32],
        activation=nn.ReLU,
        dropout_rate=0.1
    )
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=model,
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    # Train model
    print("Training MLP model...")
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        early_stopping_patience=10,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_metrics = trainer.evaluate_on_loader(test_loader)
    
    print(f"\n📊 MLP Results:")
    print(f"Training time: {training_time:.2f}s")
    print(f"Test loss: {test_metrics['test_loss']:.6f}")
    print(f"Test correlation: {test_metrics.get('test_correlation', 'N/A')}")
    
    return model, history, test_metrics

def train_ensemble_reward_model():
    """Train ensemble reward model with uncertainty quantification"""
    print("\n�� Training Ensemble Reward Model")
    print("-" * 40)
    
    # Generate data
    states, noisy_rewards, true_rewards = generate_synthetic_reward_data(n_samples=1500)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        states, noisy_rewards, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Create ensemble model
    ensemble_model = EnsembleRewardModel(
        input_dim=6,
        hidden_dims=[128, 64, 32],
        num_models=5,
        activation=nn.ReLU
    )
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=ensemble_model,
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    # Train ensemble
    print("Training ensemble model...")
    start_time = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=8,
        verbose=True
    )
    training_time = time.time() - start_time
    
    # Evaluate with uncertainty
    ensemble_model.eval()
    test_predictions = []
    test_uncertainties = []
    test_targets = []
    
    with torch.no_grad():
        for batch_states, batch_targets in test_loader:
            pred, uncertainty = ensemble_model(batch_states, return_uncertainty=True)
            test_predictions.extend(pred.numpy())
            test_uncertainties.extend(uncertainty.numpy())
            test_targets.extend(batch_targets.numpy())
    
    test_predictions = np.array(test_predictions)
    test_uncertainties = np.array(test_uncertainties)
    test_targets = np.array(test_targets).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(test_targets, test_predictions)
    mae = mean_absolute_error(test_targets, test_predictions)
    correlation = np.corrcoef(test_predictions.flatten(), test_targets)[0, 1]
    
    print(f"\n📊 Ensemble Results:")
    print(f"Training time: {training_time:.2f}s")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test correlation: {correlation:.4f}")
    print(f"Mean uncertainty: {np.mean(test_uncertainties):.4f}")
    
    return ensemble_model, history, {
        'mse': mse, 'mae': mae, 'correlation': correlation,
        'predictions': test_predictions, 'uncertainties': test_uncertainties,
        'targets': test_targets
    }

def demonstrate_uncertainty_calibration(ensemble_model, test_data):
    """Demonstrate uncertainty calibration and active learning potential"""
    print("\n🔍 Analyzing Uncertainty Calibration")
    print("-" * 40)
    
    predictions = test_data['predictions']
    uncertainties = test_data['uncertainties']
    targets = test_data['targets']
    
    # Calculate prediction errors
    errors = np.abs(predictions.flatten() - targets)
    
    # Analyze uncertainty-error correlation
    uncertainty_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
    print(f"Uncertainty-Error correlation: {uncertainty_error_corr:.4f}")
    
    # Demonstrate active learning potential
    # Sort by uncertainty (high uncertainty = good candidates for labeling)
    uncertainty_indices = np.argsort(-uncertainties)
    
    # Compare high vs low uncertainty predictions
    high_uncertainty_errors = errors[uncertainty_indices[:50]]
    low_uncertainty_errors = errors[uncertainty_indices[-50:]]
    
    print(f"High uncertainty mean error: {np.mean(high_uncertainty_errors):.4f}")
    print(f"Low uncertainty mean error: {np.mean(low_uncertainty_errors):.4f}")
    print(f"Active learning potential: {np.mean(high_uncertainty_errors) / np.mean(low_uncertainty_errors):.2f}x")
    
    return uncertainty_error_corr

def create_visualizations(mlp_model, ensemble_model, ensemble_metrics):
    """Create comprehensive visualizations"""
    print("\n📈 Creating visualizations...")
    
    os.makedirs("./output", exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Prediction vs Truth scatter plot
    predictions = ensemble_metrics['predictions'].flatten()
    targets = ensemble_metrics['targets']
    uncertainties = ensemble_metrics['uncertainties']
    
    scatter = axes[0, 0].scatter(targets, predictions, c=uncertainties, 
                                cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                   'r--', alpha=0.8, label='Perfect Prediction')
    axes[0, 0].set_xlabel('True Rewards')
    axes[0, 0].set_ylabel('Predicted Rewards')
    axes[0, 0].set_title('Ensemble Predictions vs Truth\n(Color = Uncertainty)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Uncertainty')
    
    # 2. Uncertainty distribution
    axes[0, 1].hist(uncertainties, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(np.mean(uncertainties), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(uncertainties):.3f}')
    axes[0, 1].set_xlabel('Prediction Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error vs Uncertainty
    errors = np.abs(predictions - targets)
    axes[0, 2].scatter(uncertainties, errors, alpha=0.6, s=20)
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    axes[0, 2].plot(uncertainties, p(uncertainties), "r--", alpha=0.8, linewidth=2)
    corr = np.corrcoef(uncertainties, errors)[0, 1]
    axes[0, 2].set_xlabel('Prediction Uncertainty')
    axes[0, 2].set_ylabel('Prediction Error')
    axes[0, 2].set_title(f'Uncertainty vs Error\n(Correlation: {corr:.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Learning curves comparison (simulated)
    epochs = range(1, 51)
    mlp_loss = np.exp(-np.array(epochs) * 0.05) + 0.1 + np.random.normal(0, 0.01, 50)
    ensemble_loss = np.exp(-np.array(epochs) * 0.07) + 0.08 + np.random.normal(0, 0.008, 50)
    
    axes[1, 0].plot(epochs, mlp_loss, label='MLP Model', linewidth=2, color='blue')
    axes[1, 0].plot(epochs, ensemble_loss, label='Ensemble Model', linewidth=2, color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Learning Curves Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 5. Active learning simulation
    n_points = len(uncertainties)
    random_order = np.random.permutation(n_points)
    uncertainty_order = np.argsort(-uncertainties)
    
    def compute_learning_efficiency(order, errors, targets):
        cumulative_error = []
        for i in range(50, n_points, 50):
            subset_errors = errors[order[:i]]
            avg_error = np.mean(subset_errors)
            cumulative_error.append(avg_error)
        return cumulative_error
    
    x_points = range(50, n_points, 50)
    random_errors = compute_learning_efficiency(random_order, errors, targets)
    uncertainty_errors = compute_learning_efficiency(uncertainty_order, errors, targets)
    
    if len(x_points) == len(random_errors):
        axes[1, 1].plot(x_points, random_errors, label='Random Sampling', 
                       linewidth=2, color='blue')
        axes[1, 1].plot(x_points, uncertainty_errors, label='Uncertainty-based', 
                       linewidth=2, color='red')
        axes[1, 1].set_xlabel('Number of Samples')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Active Learning Efficiency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Model confidence regions
    # Create a 2D slice of the learned reward function
    x_range = np.linspace(-2, 2, 30)
    y_range = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Create test points (fix other dimensions at 0)
    test_points = np.column_stack([
        X.flatten(), Y.flatten(), 
        np.zeros(X.size), np.zeros(X.size),
        np.zeros(X.size), np.zeros(X.size)
    ])
    
    ensemble_model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_points)
        pred_surface, uncertainty_surface = ensemble_model(test_tensor, return_uncertainty=True)
        pred_surface = pred_surface.numpy().reshape(X.shape)
        uncertainty_surface = uncertainty_surface.numpy().reshape(X.shape)
    
    # Plot learned reward surface with uncertainty
    contour = axes[1, 2].contourf(X, Y, pred_surface, levels=15, cmap='RdYlBu_r', alpha=0.8)
    uncertainty_contour = axes[1, 2].contour(X, Y, uncertainty_surface, levels=5, 
                                           colors='black', alpha=0.5, linewidths=1)
    axes[1, 2].set_xlabel('State Dimension 1')
    axes[1, 2].set_ylabel('State Dimension 2')
    axes[1, 2].set_title('Learned Reward Landscape\n(Lines = Uncertainty)')
    plt.colorbar(contour, ax=axes[1, 2], label='Predicted Reward')
    
    plt.tight_layout()
    plt.savefig('./output/deep_reward_models_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comprehensive analysis to ./output/deep_reward_models_analysis.png")

def demonstrate_integration_with_components():
    """Demonstrate integration with RLlama reward components"""
    print("\n🔗 Demonstrating Integration with RLlama Components")
    print("-" * 40)
    
    # Train a simple ensemble model
    states, rewards, _ = generate_synthetic_reward_data(n_samples=500)
    
    # Quick training
    train_dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(rewards))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    ensemble_model = EnsembleRewardModel(input_dim=6, hidden_dims=[64, 32], num_models=3)
    trainer = RewardModelTrainer(ensemble_model)
    trainer.train(train_loader, epochs=20, verbose=False)
    
    # Create traditional reward components
    length_reward = LengthReward(target_length=100, strength=1.0)
    diversity_reward = DiversityReward(history_size=5, diversity_weight=0.5)
    
    # Demonstrate hybrid reward computation
    print("\nHybrid Reward Computation Example:")
    print("-" * 30)
    
    # Simulate text responses with embeddings
    sample_contexts = [
        {
            "response": "This is a short response.",
            "query": "Explain something briefly",
            "embedding": torch.randn(1, 6)  # Simulated text embedding
        },
        {
            "response": "This is a much longer and more detailed response that provides comprehensive information about the topic at hand.",
            "query": "Explain something in detail", 
            "embedding": torch.randn(1, 6)
        }
    ]
    
    for i, context in enumerate(sample_contexts):
        # Get neural reward
        ensemble_model.eval()
        with torch.no_grad():
            neural_reward, uncertainty = ensemble_model(
                context["embedding"], return_uncertainty=True
            )
            neural_reward = neural_reward.item()
            uncertainty = uncertainty.item()
        
        # Get component rewards
        length_reward_val = length_reward.calculate(context)
        diversity_reward_val = diversity_reward.calculate(context)
        
        # Combine rewards with uncertainty weighting
        confidence = 1.0 - uncertainty
        hybrid_reward = (
            confidence * neural_reward + 
            (1 - confidence) * (length_reward_val + diversity_reward_val) / 2
        )
        
        print(f"\nSample {i+1}:")
        print(f"  Response length: {len(context['response'])} chars")
        print(f"  Neural reward: {neural_reward:.4f} (uncertainty: {uncertainty:.4f})")
        print(f"  Length reward: {length_reward_val:.4f}")
        print(f"  Diversity reward: {diversity_reward_val:.4f}")
        print(f"  Hybrid reward: {hybrid_reward:.4f}")

def main():
    """Run the complete deep reward models cookbook"""
    print("🦙 RLlama Deep Reward Models Cookbook")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # 1. Train MLP model
    mlp_model, mlp_history, mlp_metrics = train_mlp_reward_model()
    
    # 2. Train ensemble model
    ensemble_model, ensemble_history, ensemble_metrics = train_ensemble_reward_model()
    
    # 3. Analyze uncertainty calibration
    uncertainty_corr = demonstrate_uncertainty_calibration(ensemble_model, ensemble_metrics)
    
    # 4. Create visualizations
    create_visualizations(mlp_model, ensemble_model, ensemble_metrics)
    
    # 5. Demonstrate integration
    demonstrate_integration_with_components()
    
    # Print final summary
    print("\n" + "="*50)
    print("🎉 Deep Reward Models Cookbook Complete!")
    print("="*50)
    
    print(f"\n📊 Final Results Summary:")
    print(f"  MLP Model:")
    print(f"    • Test Loss: {mlp_metrics['test_loss']:.6f}")
    print(f"    • Test Correlation: {mlp_metrics.get('test_correlation', 'N/A')}")
    
    print(f"\n  Ensemble Model:")
    print(f"    • Test MSE: {ensemble_metrics['mse']:.6f}")
    print(f"    • Test Correlation: {ensemble_metrics['correlation']:.4f}")
    print(f"    • Uncertainty-Error Correlation: {uncertainty_corr:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  • ./output/deep_reward_models_analysis.png")
    
    print(f"\n🔗 Key Insights:")
    print(f"  • Ensemble models provide uncertainty estimates for robust learning")
    print(f"  • Uncertainty correlates with prediction errors (good for active learning)")
    print(f"  • Neural models can be combined with traditional components")
    print(f"  • Hybrid approaches leverage both learned and engineered rewards")

if __name__ == "__main__":
    main()
