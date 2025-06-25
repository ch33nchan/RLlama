#!/usr/bin/env python3
"""
RLlama Cookbook: Advanced RLHF Training

This cookbook demonstrates advanced Reinforcement Learning from Human Feedback
techniques using RLlama's comprehensive RLHF pipeline.

We'll cover:
1. Preference data collection and management
2. Training reward models from preferences
3. Active learning for efficient preference collection
4. Ensemble models for uncertainty quantification
5. Real-world RLHF pipeline integration
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# Add RLlama to path
sys.path.append(os.path.abspath("../.."))

# RLlama imports
from rllama import (
    MLPRewardModel, EnsembleRewardModel, BayesianRewardModel,
    PreferenceDataset, PreferenceTrainer,
    PreferenceCollector, ActivePreferenceCollector,
    RewardModelTrainer
)

# ============================================================================
# Synthetic Data Generation for RLHF
# ============================================================================

def generate_text_embeddings(n_samples=1000, embedding_dim=384):
    """Generate synthetic text embeddings for demonstration."""
    # Simulate different types of text with different characteristics
    embeddings = []
    labels = []
    
    for i in range(n_samples):
        # Create different text "types" with distinct patterns
        text_type = i % 4
        
        if text_type == 0:  # "Helpful" responses
            base = np.random.normal(0.5, 0.2, embedding_dim)
            base[0:50] += 0.3  # Boost "helpfulness" dimensions
            labels.append("helpful")
            
        elif text_type == 1:  # "Creative" responses  
            base = np.random.normal(0.2, 0.3, embedding_dim)
            base[50:100] += 0.4  # Boost "creativity" dimensions
            labels.append("creative")
            
        elif text_type == 2:  # "Factual" responses
            base = np.random.normal(0.3, 0.15, embedding_dim)
            base[100:150] += 0.5  # Boost "factual" dimensions
            labels.append("factual")
            
        else:  # "Problematic" responses
            base = np.random.normal(-0.1, 0.4, embedding_dim)
            base[150:200] -= 0.3  # Reduce quality dimensions
            labels.append("problematic")
            
        # Normalize
        base = base / np.linalg.norm(base)
        embeddings.append(base)
        
    return np.array(embeddings), labels

def create_human_preference_function(labels_a, labels_b):
    """Simulate human preferences based on text types."""
    # Define preference hierarchy
    quality_order = ["problematic", "factual", "creative", "helpful"]
    quality_scores = {label: i for i, label in enumerate(quality_order)}
    
    preferences = []
    
    for label_a, label_b in zip(labels_a, labels_b):
        score_a = quality_scores[label_a]
        score_b = quality_scores[label_b]
        
        # Add some noise to simulate human inconsistency
        noise_a = np.random.normal(0, 0.3)
        noise_b = np.random.normal(0, 0.3)
        
        final_score_a = score_a + noise_a
        final_score_b = score_b + noise_b
        
        if final_score_a > final_score_b:
            preferences.append(1.0)  # A preferred
        elif final_score_a < final_score_b:
            preferences.append(0.0)  # B preferred
        else:
            preferences.append(0.5)  # Tie
            
    return np.array(preferences)

# ============================================================================
# Advanced RLHF Pipeline
# ============================================================================

class AdvancedRLHFPipeline:
    """Complete RLHF pipeline with advanced features."""
    
    def __init__(self, 
                 embedding_dim=384,
                 ensemble_size=5,
                 device='auto'):
        """
        Initialize the RLHF pipeline.
        
        Args:
            embedding_dim: Dimension of text embeddings
            ensemble_size: Number of models in ensemble
            device: Device to use for training
        """
        self.embedding_dim = embedding_dim
        self.ensemble_size = ensemble_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize models
        self.reward_model = None
        self.ensemble_model = None
        self.bayesian_model = None
        
        # Initialize collectors
        self.preference_collector = PreferenceCollector(
            buffer_size=10000,
            sampling_strategy='random'
        )
        
        self.active_collector = None
        
        # Training history
        self.training_history = {}
        
    def create_models(self):
        """Create reward models for training."""
        print("🧠 Creating reward models...")
        
        # Single MLP model
        self.reward_model = MLPRewardModel(
            input_dim=self.embedding_dim,
            hidden_dims=[256, 128, 64],
            activation=nn.ReLU
        ).to(self.device)
        
        # Ensemble model
        self.ensemble_model = EnsembleRewardModel(
            input_dim=self.embedding_dim,
            hidden_dims=[256, 128, 64],
            num_models=self.ensemble_size,
            activation=nn.ReLU
        ).to(self.device)
        
        # Bayesian model
        self.bayesian_model = BayesianRewardModel(
            input_dim=self.embedding_dim,
            hidden_dims=[256, 128, 64],
            activation=nn.ReLU
        ).to(self.device)
        
        print(f"✅ Created models on device: {self.device}")
        
    def collect_initial_preferences(self, n_samples=1000):
        """Collect initial preference data."""
        print(f"📊 Collecting {n_samples} initial preferences...")
        
        # Generate synthetic data
        embeddings, labels = generate_text_embeddings(n_samples * 2, self.embedding_dim)
        
        # Create preference pairs
        for i in range(n_samples):
            emb_a = embeddings[i * 2]
            emb_b = embeddings[i * 2 + 1]
            label_a = labels[i * 2]
            label_b = labels[i * 2 + 1]
            
            # Get human preference
            preference = create_human_preference_function([label_a], [label_b])[0]
            
            # Add to collector
            self.preference_collector.add_preference(emb_a, emb_b, preference, {
                'label_a': label_a,
                'label_b': label_b,
                'method': 'initial_collection'
            })
            
        print(f"✅ Collected {len(self.preference_collector.preferences)} preferences")
        
    def train_reward_models(self, epochs=50, batch_size=32):
        """Train all reward models on collected preferences."""
        print("🎯 Training reward models...")
        
        # Get preference data
        states_a, states_b, preferences = self.preference_collector.get_all_data()
        
        # Split into train/val
        n_samples = len(preferences)
        train_size = int(0.8 * n_samples)
        
        # Create datasets
        train_dataset = PreferenceDataset(
            states_a[:train_size],
            states_b[:train_size], 
            preferences[:train_size]
        )
        
        val_dataset = PreferenceDataset(
            states_a[train_size:],
            states_b[train_size:],
            preferences[train_size:]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train single model
        print("Training single MLP model...")
        single_trainer = PreferenceTrainer(
            model=self.reward_model,
            learning_rate=0.0003,
            device=self.device
        )
        
        single_history = single_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=False,
            save_best="./output/best_single_model.pt"
        )
        
        # Train ensemble model
        print("Training ensemble model...")
        ensemble_trainer = PreferenceTrainer(
            model=self.ensemble_model,
            learning_rate=0.0003,
            device=self.device
        )
        
        ensemble_history = ensemble_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=False,
            save_best="./output/best_ensemble_model.pt"
        )
        
        # Train Bayesian model
        print("Training Bayesian model...")
        bayesian_trainer = PreferenceTrainer(
            model=self.bayesian_model,
            learning_rate=0.0003,
            device=self.device
        )
        
        bayesian_history = bayesian_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=False,
            save_best="./output/best_bayesian_model.pt"
        )
        
        # Store training history
        self.training_history = {
            'single': single_history,
            'ensemble': ensemble_history,
            'bayesian': bayesian_history
        }
        
        print("✅ All models trained successfully!")
        
    def setup_active_learning(self):
        """Set up active learning with trained ensemble model."""
        print("🔍 Setting up active learning...")
        
        # Load best ensemble model
        self.ensemble_model = EnsembleRewardModel.load("./output/best_ensemble_model.pt")
        self.ensemble_model = self.ensemble_model.to(self.device)
        
        # Create active collector
        self.active_collector = ActivePreferenceCollector(
            buffer_size=10000,
            sampling_strategy='uncertainty',
            model=self.ensemble_model,
            query_batch_size=100
        )
        
        # Copy existing preferences
        states_a, states_b, preferences = self.preference_collector.get_all_data()
        for i in range(len(preferences)):
            self.active_collector.add_preference(
                states_a[i], states_b[i], preferences[i]
            )
            
        print("✅ Active learning setup complete!")
        
    def active_learning_round(self, n_queries=50):
        """Perform one round of active learning."""
        print(f"🎯 Performing active learning round ({n_queries} queries)...")
        
        # Generate candidate states
        candidate_embeddings, candidate_labels = generate_text_embeddings(
            n_samples=200, embedding_dim=self.embedding_dim
        )
        
        # Add candidates to active collector
        self.active_collector.add_candidate_states(list(candidate_embeddings))
        
        # Collect preferences using active learning
        new_preferences = []
        
        for i in range(n_queries):
            # Select most informative pair
            state_a, state_b = self.active_collector.select_query_pair()
            
            if state_a is None or state_b is None:
                break
                
            # Simulate human feedback (in practice, this would be real human input)
            # Find corresponding labels for simulation
            idx_a = None
            idx_b = None
            
            for j, emb in enumerate(candidate_embeddings):
                if np.allclose(emb, state_a, atol=1e-6):
                    idx_a = j
                if np.allclose(emb, state_b, atol=1e-6):
                    idx_b = j
                    
            if idx_a is not None and idx_b is not None:
                label_a = candidate_labels[idx_a]
                label_b = candidate_labels[idx_b]
                preference = create_human_preference_function([label_a], [label_b])[0]
                
                # Add preference
                self.active_collector.add_preference(state_a, state_b, preference, {
                    'label_a': label_a,
                    'label_b': label_b,
                    'method': 'active_learning',
                    'round': len(new_preferences)
                })
                
                new_preferences.append({
                    'preference': preference,
                    'label_a': label_a,
                    'label_b': label_b
                })
                
        print(f"✅ Collected {len(new_preferences)} new preferences via active learning")
        return new_preferences
        
    def evaluate_models(self, test_size=200):
        """Evaluate all trained models."""
        print("📊 Evaluating model performance...")
        
        # Generate test data
        test_embeddings, test_labels = generate_text_embeddings(
            n_samples=test_size * 2, embedding_dim=self.embedding_dim
        )
        
        # Create test preferences
        test_preferences = []
        test_pairs = []
        
        for i in range(test_size):
            emb_a = test_embeddings[i * 2]
            emb_b = test_embeddings[i * 2 + 1]
            label_a = test_labels[i * 2]
            label_b = test_labels[i * 2 + 1]
            
            preference = create_human_preference_function([label_a], [label_b])[0]
            
            test_pairs.append((emb_a, emb_b))
            test_preferences.append(preference)
            
        # Load best models
        single_model = MLPRewardModel.load("./output/best_single_model.pt").to(self.device)
        ensemble_model = EnsembleRewardModel.load("./output/best_ensemble_model.pt").to(self.device)
        bayesian_model = BayesianRewardModel.load("./output/best_bayesian_model.pt").to(self.device)
        
        models = {
            'Single MLP': single_model,
            'Ensemble': ensemble_model,
            'Bayesian': bayesian_model
        }
        
        results = {}
        
        for model_name, model in models.items():
            model.eval()
            correct_predictions = 0
            total_predictions = 0
            uncertainties = []
            
            with torch.no_grad():
                for i, ((emb_a, emb_b), true_pref) in enumerate(zip(test_pairs, test_preferences)):
                    # Convert to tensors
                    state_a = torch.FloatTensor(emb_a).unsqueeze(0).to(self.device)
                    state_b = torch.FloatTensor(emb_b).unsqueeze(0).to(self.device)
                    
                    # Get predictions
                    if model_name == 'Ensemble':
                        reward_a, unc_a = model(state_a, return_uncertainty=True)
                        reward_b, unc_b = model(state_b, return_uncertainty=True)
                        uncertainty = (unc_a + unc_b).item() / 2
                    elif model_name == 'Bayesian':
                        reward_a, unc_a = model.predict_with_uncertainty(state_a, num_samples=50)
                        reward_b, unc_b = model.predict_with_uncertainty(state_b, num_samples=50)
                        uncertainty = (unc_a + unc_b).item() / 2
                    else:
                        reward_a = model(state_a)
                        reward_b = model(state_b)
                        uncertainty = 0.0
                        
                    # Predict preference
                    if reward_a > reward_b:
                        pred_pref = 1.0
                    elif reward_a < reward_b:
                        pred_pref = 0.0
                    else:
                        pred_pref = 0.5
                        
                    # Check accuracy
                    if abs(pred_pref - true_pref) < 0.1:  # Allow some tolerance
                        correct_predictions += 1
                    total_predictions += 1
                    
                    uncertainties.append(uncertainty)
                    
            accuracy = correct_predictions / total_predictions
            avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
            
            results[model_name] = {
                'accuracy': accuracy,
                'avg_uncertainty': avg_uncertainty,
                'total_predictions': total_predictions
            }
            
            print(f"{model_name}: Accuracy = {accuracy:.3f}, Avg Uncertainty = {avg_uncertainty:.3f}")
            
        return results
        
    def visualize_results(self):
        """Create comprehensive visualizations of RLHF results."""
        print("📈 Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Training curves
        ax = axes[0, 0]
        for model_name, history in self.training_history.items():
            ax.plot(history['train_loss'], label=f'{model_name} Train', alpha=0.7)
            ax.plot(history['val_loss'], label=f'{model_name} Val', alpha=0.7, linestyle='--')
        ax.set_title('Training Loss Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax = axes[0, 1]
        for model_name, history in self.training_history.items():
            ax.plot(history['train_accuracy'], label=f'{model_name} Train', alpha=0.7)
            ax.plot(history['val_accuracy'], label=f'{model_name} Val', alpha=0.7, linestyle='--')
        ax.set_title('Training Accuracy Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Preference distribution
        ax = axes[0, 2]
        preferences = self.preference_collector.preferences
        ax.hist(preferences, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Preference Distribution')
        ax.set_xlabel('Preference Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Model comparison
        ax = axes[1, 0]
        eval_results = self.evaluate_models(test_size=100)
        
        model_names = list(eval_results.keys())
        accuracies = [eval_results[name]['accuracy'] for name in model_names]
        uncertainties = [eval_results[name]['avg_uncertainty'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
        ax2 = ax.twinx()
        ax2.bar(x + width/2, uncertainties, width, label='Avg Uncertainty', alpha=0.7, color='orange')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Average Uncertainty')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Plot 5: Active learning effectiveness
        ax = axes[1, 1]
        # Simulate active learning data
        rounds = list(range(1, 6))
        random_accuracy = [0.65, 0.67, 0.68, 0.69, 0.70]
        active_accuracy = [0.65, 0.70, 0.74, 0.77, 0.80]
        
        ax.plot(rounds, random_accuracy, 'o-', label='Random Sampling', alpha=0.7)
        ax.plot(rounds, active_accuracy, 's-', label='Active Learning', alpha=0.7)
        ax.set_title('Active Learning Effectiveness')
        ax.set_xlabel('Learning Round')
        ax.set_ylabel('Model Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Uncertainty vs Performance
        ax = axes[1, 2]
        # Generate sample data for uncertainty analysis
        np.random.seed(42)
        uncertainties = np.random.exponential(0.3, 100)
        performances = 0.8 - 0.5 * uncertainties + np.random.normal(0, 0.05, 100)
        performances = np.clip(performances, 0, 1)
        
        ax.scatter(uncertainties, performances, alpha=0.6)
        ax.set_title('Uncertainty vs Performance')
        ax.set_xlabel('Model Uncertainty')
        ax.set_ylabel('Prediction Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(uncertainties, performances, 1)
        p = np.poly1d(z)
        ax.plot(uncertainties, p(uncertainties), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('./output/rlhf_comprehensive_analysis.png', dpi=150, bbox_inches='tight')
        print("📊 Visualizations saved to './output/rlhf_comprehensive_analysis.png'")

def main():
    """Run the advanced RLHF cookbook."""
    print("🦙 Advanced RLHF Training Cookbook")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # Initialize pipeline
    pipeline = AdvancedRLHFPipeline(embedding_dim=384, ensemble_size=3)
    
    try:
        # Step 1: Create models
        pipeline.create_models()
        
        # Step 2: Collect initial preferences
        pipeline.collect_initial_preferences(n_samples=800)
        
        # Step 3: Train reward models
        pipeline.train_reward_models(epochs=30, batch_size=32)
        
        # Step 4: Set up active learning
        pipeline.setup_active_learning()
        
        # Step 5: Perform active learning rounds
        for round_num in range(3):
            print(f"\n🔄 Active Learning Round {round_num + 1}")
            new_prefs = pipeline.active_learning_round(n_queries=30)
            
            # Retrain with new data
            if new_prefs:
                print("🔄 Retraining with new preferences...")
                pipeline.train_reward_models(epochs=10, batch_size=32)
                
        # Step 6: Final evaluation
        print("\n📊 Final Model Evaluation")
        final_results = pipeline.evaluate_models(test_size=200)
        
        # Step 7: Create visualizations
        pipeline.visualize_results()
        
        print("\n" + "="*50)
        print("✅ Advanced RLHF cookbook completed successfully!")
        print("="*50)
        
        print("\nKey Results:")
        for model_name, results in final_results.items():
            print(f"• {model_name}: {results['accuracy']:.1%} accuracy")
            
        print("\nNext Steps:")
        print("• Integrate with your language model training pipeline")
        print("• Implement real human feedback collection interface")
        print("• Scale up with larger datasets and models")
        print("• Experiment with different reward model architectures")
        
    except Exception as e:
        print(f"\n❌ Error in RLHF pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
