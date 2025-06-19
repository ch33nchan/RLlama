# rllama/rlhf/preference.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import time
from tqdm import tqdm

from ..models import BaseRewardModel

class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset for preference-based learning."""
    
    def __init__(self, 
                states_a: np.ndarray, 
                states_b: np.ndarray, 
                preferences: np.ndarray):
        """
        Initialize preference dataset.
        
        Args:
            states_a: First state in each comparison
            states_b: Second state in each comparison
            preferences: 1.0 if A > B, 0.5 if A = B, 0.0 if A < B
        """
        self.states_a = states_a
        self.states_b = states_b
        self.preferences = preferences
        
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.preferences)
        
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get dataset item."""
        return self.states_a[idx], self.states_b[idx], self.preferences[idx]


class PreferenceTrainer:
    """Trainer for preference-based learning."""
    
    def __init__(self, 
                model: BaseRewardModel,
                learning_rate: float = 0.0003,
                weight_decay: float = 0.00001,
                temperature: float = 1.0,
                device: str = 'auto'):
        """
        Initialize preference trainer.
        
        Args:
            model: Reward model to train
            learning_rate: Learning rate for optimization
            weight_decay: L2 regularization
            temperature: Temperature for Bradley-Terry model
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model = model
        self.temperature = temperature
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': 0
        }
        
    def preference_loss(self, 
                       rewards_a: torch.Tensor, 
                       rewards_b: torch.Tensor, 
                       preferences: torch.Tensor) -> torch.Tensor:
        """
        Calculate Bradley-Terry preference loss.
        
        Args:
            rewards_a: Predicted rewards for first option
            rewards_b: Predicted rewards for second option
            preferences: Ground truth preferences (1.0 if A > B, etc.)
            
        Returns:
            Loss value
        """
        # Calculate logits from rewards
        logits = (rewards_a - rewards_b) / self.temperature
        
        # Calculate probabilities with sigmoid
        probs = torch.sigmoid(logits)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(probs, preferences)
        
        return loss
        
    def train_step(self, 
                  states_a: torch.Tensor, 
                  states_b: torch.Tensor, 
                  preferences: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            states_a: Batch of first states
            states_b: Batch of second states
            preferences: Ground truth preferences
            
        Returns:
            Dict with loss and accuracy
        """
        # Move data to device
        states_a = states_a.to(self.device)
        states_b = states_b.to(self.device)
        preferences = preferences.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        rewards_a = self.model(states_a)
        rewards_b = self.model(states_b)
        
        # Calculate loss
        loss = self.preference_loss(rewards_a, rewards_b, preferences)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            pred_prefs = (rewards_a > rewards_b).float()
            # Consider ties as 0.5
            pred_prefs[torch.isclose(rewards_a, rewards_b)] = 0.5
            accuracy = torch.mean((pred_prefs == preferences).float()).item()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
    def evaluate(self, 
                states_a: torch.Tensor, 
                states_b: torch.Tensor, 
                preferences: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            states_a: Batch of first states
            states_b: Batch of second states
            preferences: Ground truth preferences
            
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        # Move data to device
        states_a = states_a.to(self.device)
        states_b = states_b.to(self.device)
        preferences = preferences.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            rewards_a = self.model(states_a)
            rewards_b = self.model(states_b)
            
            # Calculate loss
            loss = self.preference_loss(rewards_a, rewards_b, preferences).item()
            
            # Calculate accuracy
            pred_prefs = (rewards_a > rewards_b).float()
            # Consider ties as 0.5
            pred_prefs[torch.isclose(rewards_a, rewards_b)] = 0.5
            accuracy = torch.mean((pred_prefs == preferences).float()).item()
            
        self.model.train()
        
        return {
            'val_loss': loss,
            'val_accuracy': accuracy
        }
        
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: Optional[torch.utils.data.DataLoader] = None,
             epochs: int = 10,
             early_stopping_patience: int = 5,
             verbose: bool = True,
             save_best: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            save_best: Path to save best model, or None
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # Track epochs
        start_epoch = self.history['epochs']
        self.history['epochs'] += epochs
        
        for epoch in range(start_epoch, start_epoch + epochs):
            # Training loop
            self.model.train()
            train_losses = []
            train_accuracies = []
            
            pbar = tqdm(train_loader, disable=not verbose)
            for batch in pbar:
                # Extract batch data
                states_a, states_b, preferences = batch
                
                # Training step
                metrics = self.train_step(states_a, states_b, preferences)
                train_losses.append(metrics['loss'])
                train_accuracies.append(metrics['accuracy'])
                
                # Update progress bar
                if verbose:
                    pbar.set_description(
                        f"Epoch {epoch+1}/{start_epoch+epochs} | "
                        f"Loss: {metrics['loss']:.6f} | "
                        f"Acc: {metrics['accuracy']:.4f}"
                    )
                    
            # Calculate average training metrics
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_accuracy'].append(avg_train_acc)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate_on_loader(val_loader)
                val_loss = val_metrics['val_loss']
                val_acc = val_metrics['val_accuracy']
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if save_best is not None:
                        self.model.save(save_best)
                else:
                    patience_counter += 1
                    
                # Print epoch summary
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{start_epoch+epochs} | "
                        f"Train Loss: {avg_train_loss:.6f} | "
                        f"Train Acc: {avg_train_acc:.4f} | "
                        f"Val Loss: {val_loss:.6f} | "
                        f"Val Acc: {val_acc:.4f}"
                    )
                    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # Print epoch summary without validation
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{start_epoch+epochs} | "
                        f"Train Loss: {avg_train_loss:.6f} | "
                        f"Train Acc: {avg_train_acc:.4f}"
                    )
        
        # Report training time
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            
        return self.history
        
    def evaluate_on_loader(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a full data loader.
        
        Args:
            data_loader: DataLoader with validation data
            
        Returns:
            Dict with validation metrics
        """
        all_losses = []
        all_accuracies = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                # Extract batch data
                states_a, states_b, preferences = batch
                
                # Move data to device
                states_a = states_a.to(self.device)
                states_b = states_b.to(self.device)
                preferences = preferences.to(self.device)
                
                # Forward pass
                rewards_a = self.model(states_a)
                rewards_b = self.model(states_b)
                
                # Calculate loss
                loss = self.preference_loss(rewards_a, rewards_b, preferences).item()
                
                # Calculate accuracy
                pred_prefs = (rewards_a > rewards_b).float()
                # Consider ties as 0.5
                pred_prefs[torch.isclose(rewards_a, rewards_b)] = 0.5
                accuracy = torch.mean((pred_prefs == preferences).float()).item()
                
                all_losses.append(loss)
                all_accuracies.append(accuracy)
                
        # Calculate averages
        avg_loss = np.mean(all_losses)
        avg_accuracy = np.mean(all_accuracies)
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy
        }
        
        return metrics
