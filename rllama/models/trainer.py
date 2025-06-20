

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import time
from tqdm import tqdm

from .base import BaseRewardModel

class RewardModelTrainer:
    """Trainer for reward models."""
    
    def __init__(self, 
                model: BaseRewardModel,
                learning_rate: float = 0.001,
                weight_decay: float = 0.0001,
                device: str = 'auto'):
        """
        Initialize the reward model trainer.
        
        Args:
            model: The reward model to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay (L2 regularization)
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model = model
        
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
        
        # Loss function (Mean Squared Error)
        self.loss_fn = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0
        }
        
    def train_step(self, 
                  states: torch.Tensor, 
                  rewards: torch.Tensor,
                  actions: Optional[torch.Tensor] = None) -> float:
        """
        Perform a single training step.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            rewards: Ground truth rewards (batch_size, 1)
            actions: Optional batch of actions (batch_size, action_dim)
            
        Returns:
            Training loss value
        """
        # Move data to device
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        
        if actions is not None:
            actions = actions.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_rewards = self.model(states, actions)
        
        # Calculate loss
        loss = self.loss_fn(pred_rewards, rewards)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def evaluate(self, 
                states: torch.Tensor, 
                rewards: torch.Tensor,
                actions: Optional[torch.Tensor] = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation data.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            rewards: Ground truth rewards (batch_size, 1)
            actions: Optional batch of actions (batch_size, action_dim)
            
        Returns:
            Tuple of (validation loss, metrics dict)
        """
        self.model.eval()
        
        # Move data to device
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        
        if actions is not None:
            actions = actions.to(self.device)
            
        with torch.no_grad():
            # Forward pass
            pred_rewards = self.model(states, actions)
            
            # Calculate loss
            val_loss = self.loss_fn(pred_rewards, rewards).item()
            
            # Additional metrics
            mae = torch.mean(torch.abs(pred_rewards - rewards)).item()
            
            # Correlation coefficient
            pred_flat = pred_rewards.flatten()
            target_flat = rewards.flatten()
            vx = pred_flat - torch.mean(pred_flat)
            vy = target_flat - torch.mean(target_flat)
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
            
        self.model.train()
        
        metrics = {
            'val_loss': val_loss,
            'val_mae': mae,
            'val_correlation': corr.item()
        }
        
        return val_loss, metrics
        
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
            
            pbar = tqdm(train_loader, disable=not verbose)
            for batch in pbar:
                # Extract batch data
                if len(batch) == 2:  # (states, rewards)
                    states, rewards = batch
                    actions = None
                else:  # (states, actions, rewards)
                    states, actions, rewards = batch
                    
                # Training step
                loss = self.train_step(states, rewards, actions)
                train_losses.append(loss)
                
                # Update progress bar
                if verbose:
                    pbar.set_description(f"Epoch {epoch+1}/{start_epoch+epochs} | Loss: {loss:.6f}")
                    
            # Calculate average training loss
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate_on_loader(val_loader)
                self.history['val_loss'].append(val_loss)
                
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
                    print(f"Epoch {epoch+1}/{start_epoch+epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Corr: {val_metrics['val_correlation']:.6f}")
                    
                # Check early stopping
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                # Print epoch summary without validation
                if verbose:
                    print(f"Epoch {epoch+1}/{start_epoch+epochs} | Train Loss: {avg_train_loss:.6f}")
        
        # Report training time
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            
        return self.history
        
    def evaluate_on_loader(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on a full data loader.
        
        Args:
            data_loader: DataLoader with validation data
            
        Returns:
            Tuple of (average validation loss, metrics dict)
        """
        all_losses = []
        all_mae = []
        all_corr = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                # Extract batch data
                if len(batch) == 2:  # (states, rewards)
                    states, rewards = batch
                    actions = None
                else:  # (states, actions, rewards)
                    states, actions, rewards = batch
                    
                # Move data to device
                states = states.to(self.device)
                rewards = rewards.to(self.device)
                
                if actions is not None:
                    actions = actions.to(self.device)
                
                # Forward pass
                pred_rewards = self.model(states, actions)
                
                # Calculate metrics
                loss = self.loss_fn(pred_rewards, rewards).item()
                mae = torch.mean(torch.abs(pred_rewards - rewards)).item()
                
                # Correlation
                pred_flat = pred_rewards.flatten()
                target_flat = rewards.flatten()
                vx = pred_flat - torch.mean(pred_flat)
                vy = target_flat - torch.mean(target_flat)
                corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                
                all_losses.append(loss)
                all_mae.append(mae)
                all_corr.append(corr.item())
                
        # Calculate averages
        avg_loss = np.mean(all_losses)
        avg_mae = np.mean(all_mae)
        avg_corr = np.mean(all_corr)
        
        metrics = {
            'val_loss': avg_loss,
            'val_mae': avg_mae,
            'val_correlation': avg_corr
        }
        
        return avg_loss, metrics
