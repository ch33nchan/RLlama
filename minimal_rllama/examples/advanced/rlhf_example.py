#!/usr/bin/env python3
"""
Example demonstrating Reinforcement Learning from Human Feedback (RLHF) in RLlama.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

# Add parent directory to path
sys.path.append(os.path.abspath("../.."))

from rllama.models import MLPRewardModel
from rllama.rlhf import PreferenceDataset, PreferenceTrainer, PreferenceCollector, ActivePreferenceCollector

def generate_synthetic_preferences(n_samples=1000, state_dim=4, noise_level=0.1):
    """Generate synthetic preference data based on a ground truth reward function."""
    
    # Generate random states
    states = np.random.uniform(-1, 1, (n_samples * 2, state_dim))
    
    # Ground truth reward function: weighted sum with nonlinearity
    true_weights = np.array([0.7, -0.3, 0.5, -0.2])
    
    def true_reward(state):
        linear = np.dot(state, true_weights)
        nonlinear = np.sin(2.0 * state[0]) + 0.5 * state[1]**2 - 0.2 * state[2] * state[3]
        return linear + nonlinear
    
    # Calculate true rewards for all states
    true_rewards = np.array([true_reward(state) for state in states])
    
    # Add noise to rewards
    noisy_rewards = true_rewards + np.random.normal(0, noise_level, len(true_rewards))
    
    # Create preference pairs
    states_a = []
    states_b = []
    preferences = []
    
    for i in range(n_samples):
        # Select two random states
        idx_a = i * 2
        idx_b = i * 2 + 1
        
        state_a = states[idx_a]
        state_b = states[idx_b]
        
        # Determine preference based on noisy rewards
        if noisy_rewards[idx_a] > noisy_rewards[idx_b]:
            pref = 1.0  # A is preferred
        elif noisy_rewards[idx_a] < noisy_rewards[idx_b]:
            pref = 0.0  # B is preferred
        else:
            pref = 0.5  # Tie
            
        states_a.append(state_a)
        states_b.append(state_b)
        preferences.append(pref)
    
    # Convert to numpy arrays
    return np.array(states_a), np.array(states_b), np.array(preferences), true_weights

def main():
    """Run RLHF example"""
    print("RLlama RLHF Example")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic preference data
    print("\n1. Generating synthetic preference data")
    print("-" * 40)
    
    n_samples = 1000
    state_dim = 4
    states_a, states_b, preferences, true_weights = generate_synthetic_preferences(
        n_samples=n_samples, state_dim=state_dim)
    
    print(f"Generated {n_samples} preference pairs with state dimension {state_dim}")
    print(f"True reward weights: {true_weights}")
    
    # Create preference collectors
    print("\n2. Setting up preference collectors")
    print("-" * 40)
    
    collector = PreferenceCollector(buffer_size=10000, sampling_strategy='random')
    
    # Add preferences to collector
    for i in range(len(preferences)):
        collector.add_preference(states_a[i], states_b[i], preferences[i])
        
    print(f"Added {len(preferences)} preferences to collector")
    
    # Split into train/validation sets
    all_states_a, all_states_b, all_prefs = collector.get_all_data()
    
    train_size = int(0.8 * n_samples)
    
    train_states_a = all_states_a[:train_size]
    train_states_b = all_states_b[:train_size]
    train_prefs = all_prefs[:train_size]
    
    val_states_a = all_states_a[train_size:]
    val_states_b = all_states_b[train_size:]
    val_prefs = all_prefs[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = PreferenceDataset(train_states_a, train_states_b, train_prefs)
    val_dataset = PreferenceDataset(val_states_a, val_states_b, val_prefs)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create and train reward model from preferences
    print("\n3. Training reward model from preferences")
    print("-" * 40)
    
    # Create model
    reward_model = MLPRewardModel(
        input_dim=state_dim,
        hidden_dims=[64, 64],
        activation=nn.ReLU
    )
    
    # Create trainer
    trainer = PreferenceTrainer(
        model=reward_model,
        learning_rate=0.001,
        temperature=0.5
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=5,
        verbose=True,
        save_best="./output/best_preference_model.pt"
    )
    
    # Load best model
    best_model = MLPRewardModel.load("./output/best_preference_model.pt")
    
    # Demonstrate active learning
    print("\n4. Demonstrating active preference learning")
    print("-" * 40)
    
    # Create active collector with the trained model
    active_collector = ActivePreferenceCollector(
        buffer_size=10000,
        sampling_strategy='uncertainty',
        model=best_model
    )
    
    # Generate new candidate states
    n_candidates = 200
    candidate_states = np.random.uniform(-1, 1, (n_candidates, state_dim))
    active_collector.add_candidate_states(list(candidate_states))
    
    print(f"Generated {n_candidates} candidate states for active learning")
    
    # Select most informative pairs for querying
    n_queries = 10
    print(f"Selecting {n_queries} most informative pairs to query:")
    
    for i in range(n_queries):
        # Select pair
        state_a, state_b = active_collector.select_query_pair()
        
        # Simulate human feedback using ground truth
        def true_reward(state):
            linear = np.dot(state, true_weights)
            nonlinear = np.sin(2.0 * state[0]) + 0.5 * state[1]**2 - 0.2 * state[2] * state[3]
            return linear + nonlinear
            
        reward_a = true_reward(state_a)
        reward_b = true_reward(state_b)
        
        if reward_a > reward_b:
            pref = 1.0  # A is preferred
        elif reward_a < reward_b:
            pref = 0.0  # B is preferred
        else:
            pref = 0.5  # Tie
            
        # Add preference to collector
        active_collector.add_preference(state_a, state_b, pref)
        
        print(f"  Query {i+1}: True rewards: A={reward_a:.4f}, B={reward_b:.4f}, Preference: {'A' if pref == 1.0 else 'B' if pref == 0.0 else 'Tie'}")
    
    # Evaluate model on random test states
    print("\n5. Evaluating learned reward model")
    print("-" * 40)
    
    # Generate test states
    n_test = 100
    test_states = np.random.uniform(-1, 1, (n_test, state_dim))
    
    # Calculate true and predicted rewards
    true_rewards = np.array([true_reward(state) for state in test_states])
    
    # Predict with model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = best_model.to(device)
    best_model.eval()
    
    with torch.no_grad():
        test_tensor = torch.FloatTensor(test_states).to(device)
        pred_rewards = best_model(test_tensor).cpu().numpy().flatten()
    
    # Calculate correlation
    correlation = np.corrcoef(true_rewards, pred_rewards)[0, 1]
    print(f"Correlation between true and predicted rewards: {correlation:.4f}")
    
    # Visualize results
    print("\n6. Visualizing results")
    print("-" * 40)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Preference Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("./output/rlhf_training_history.png")
    
    # Plot true vs predicted rewards
    plt.figure(figsize=(10, 6))
    plt.scatter(true_rewards, pred_rewards, alpha=0.6)
    plt.plot([true_rewards.min(), true_rewards.max()], 
             [true_rewards.min(), true_rewards.max()], 
             'r--', alpha=0.5)
    plt.xlabel('True Reward')
    plt.ylabel('Predicted Reward')
    plt.title(f'True vs Predicted Rewards (Correlation: {correlation:.4f})')
    plt.grid(True, alpha=0.3)
    plt.savefig("./output/rlhf_reward_correlation.png")
    
    print("\n✅ RLHF example completed successfully!")
    print(f"Results saved to ./output/")

if __name__ == "__main__":
    main()
