#!/usr/bin/env python3
"""
Example demonstrating neural network reward models in RLlama.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama.models import MLPRewardModel, EnsembleRewardModel, RewardModelTrainer

def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """Generate synthetic data for reward modeling."""
    # Generate random states
    states = np.random.uniform(-1, 1, (n_samples, 4))
    
    # True reward function (quadratic with some nonlinearities)
    true_rewards = (
        -0.5 * states[:, 0]**2  # Quadratic term
        + np.sin(3 * states[:, 1])  # Nonlinear term
        - 0.3 * states[:, 2] * states[:, 3]  # Interaction term
        + 0.2 * states[:, 3]  # Linear term
    )
    
    # Add noise
    noisy_rewards = true_rewards + np.random.normal(0, noise_level, n_samples)
    
    # Reshape rewards to match model output shape
    noisy_rewards = noisy_rewards.reshape(-1, 1)
    
    return states, noisy_rewards, true_rewards.reshape(-1, 1)

def main():
    """Run deep reward model example"""
    print("RLlama Deep Reward Model Example")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data")
    print("-" * 40)
    
    n_samples = 2000
    states, noisy_rewards, true_rewards = generate_synthetic_data(n_samples=n_samples)
    
    print(f"Generated {n_samples} samples with state dimension {states.shape[1]}")
    
    # Split into train/validation sets
    train_size = int(0.8 * n_samples)
    
    train_states = states[:train_size]
    train_rewards = noisy_rewards[:train_size]
    
    val_states = states[train_size:]
    val_rewards = noisy_rewards[train_size:]
    val_true_rewards = true_rewards[train_size:]
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_rewards)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_states),
        torch.FloatTensor(val_rewards)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create and train a simple MLP reward model
    print("\n2. Training MLP Reward Model")
    print("-" * 40)
    
    mlp_model = MLPRewardModel(
        input_dim=states.shape[1],
        hidden_dims=[64, 32],
        activation=nn.ReLU
    )
    
    mlp_trainer = RewardModelTrainer(
        model=mlp_model,
        learning_rate=0.001
    )
    
    # Train the model
    mlp_history = mlp_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=5,
        verbose=True,
        save_best="./output/best_mlp_reward_model.pt"
    )
    
    # Create and train an ensemble reward model
    print("\n3. Training Ensemble Reward Model")
    print("-" * 40)
    
    ensemble_model = EnsembleRewardModel(
        input_dim=states.shape[1],
        hidden_dims=[64, 32],
        num_models=5
    )
    
    ensemble_trainer = RewardModelTrainer(
        model=ensemble_model,
        learning_rate=0.001
    )
    
    # Train the model
    ensemble_history = ensemble_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=5,
        verbose=True,
        save_best="./output/best_ensemble_reward_model.pt"
    )
    
    # Evaluate and compare models
    print("\n4. Evaluating Models")
    print("-" * 40)
    
    # Load best models
    best_mlp = MLPRewardModel.load("./output/best_mlp_reward_model.pt")
    best_ensemble = EnsembleRewardModel.load("./output/best_ensemble_reward_model.pt")
    
    best_mlp.eval()
    best_ensemble.eval()
    
    # Move to the right device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_mlp = best_mlp.to(device)
    best_ensemble = best_ensemble.to(device)
    
    # Evaluate on validation set
    val_states_tensor = torch.FloatTensor(val_states).to(device)
    
    with torch.no_grad():
        # Get predictions
        mlp_preds = best_mlp(val_states_tensor).cpu().numpy()
        
        # Get ensemble predictions and uncertainty
        ensemble_preds, ensemble_uncertainty = best_ensemble(
            val_states_tensor, return_uncertainty=True)
        ensemble_preds = ensemble_preds.cpu().numpy()
        ensemble_uncertainty = ensemble_uncertainty.cpu().numpy()
    
    # Calculate metrics
    mlp_mse = np.mean((mlp_preds - val_true_rewards) ** 2)
    ensemble_mse = np.mean((ensemble_preds - val_true_rewards) ** 2)
    
    print(f"MLP Model MSE vs True Rewards: {mlp_mse:.6f}")
    print(f"Ensemble Model MSE vs True Rewards: {ensemble_mse:.6f}")
    
    # Visualize results
    print("\n5. Visualizing Results")
    print("-" * 40)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_history['train_loss'], label='MLP Train Loss')
    plt.plot(mlp_history['val_loss'], label='MLP Val Loss')
    plt.plot(ensemble_history['train_loss'], label='Ensemble Train Loss')
    plt.plot(ensemble_history['val_loss'], label='Ensemble Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("./output/training_curves.png")
    
    # Sample 100 random points for visualization
    sample_indices = np.random.choice(len(val_states), 100, replace=False)
    
    # Scatter plot of predictions vs. true rewards
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(val_true_rewards[sample_indices], mlp_preds[sample_indices], alpha=0.6)
    plt.plot([-2, 2], [-2, 2], 'r--')  # Perfect prediction line
    plt.xlabel('True Reward')
    plt.ylabel('MLP Predicted Reward')
    plt.title('MLP Model Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(val_true_rewards[sample_indices], ensemble_preds[sample_indices], alpha=0.6)
    plt.errorbar(
        val_true_rewards[sample_indices].flatten(),
        ensemble_preds[sample_indices].flatten(),
        yerr=ensemble_uncertainty[sample_indices].flatten(),
        fmt='o',
        alpha=0.3,
        ecolor='red',
        capsize=0
    )
    plt.plot([-2, 2], [-2, 2], 'r--')  # Perfect prediction line
    plt.xlabel('True Reward')
    plt.ylabel('Ensemble Predicted Reward')
    plt.title('Ensemble Model Predictions with Uncertainty')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./output/prediction_comparison.png")
    
    # Visualize ensemble uncertainty
    plt.figure(figsize=(10, 6))
    
    # Sort by uncertainty for better visualization
    sorted_indices = np.argsort(ensemble_uncertainty.flatten())
    sorted_preds = ensemble_preds.flatten()[sorted_indices]
    sorted_true = val_true_rewards.flatten()[sorted_indices]
    sorted_uncertainty = ensemble_uncertainty.flatten()[sorted_indices]
    
    plt.errorbar(
        np.arange(len(sorted_indices)),
        sorted_preds,
        yerr=sorted_uncertainty,
        fmt='o',
        alpha=0.5,
        ecolor='red',
        capsize=0
    )
    plt.scatter(np.arange(len(sorted_indices)), sorted_true, alpha=0.5, color='blue', label='True Reward')
    plt.xlabel('Sample Index (Sorted by Uncertainty)')
    plt.ylabel('Reward')
    plt.title('Ensemble Predictions with Uncertainty vs. True Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("./output/uncertainty_visualization.png")
    
    print("\n✅ Deep reward model example completed successfully!")
    print(f"Results saved to ./output/")

if __name__ == "__main__":
    main()
