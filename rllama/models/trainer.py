#!/usr/bin/env python3
"""
Advanced trainer for reward models in RLlama framework.
Provides comprehensive training capabilities with sophisticated features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import time
import logging
from tqdm import tqdm
from collections import defaultdict
import warnings

from .base import BaseRewardModel

class RewardModelTrainer:
    """
    Advanced trainer for reward models with comprehensive features.
    
    Features:
    - Multiple loss functions and optimizers
    - Advanced learning rate scheduling
    - Gradient clipping and regularization
    - Comprehensive metrics tracking
    - Early stopping with multiple criteria
    - Model checkpointing and resuming
    - Mixed precision training support
    """
    
    def __init__(self, 
                 model: BaseRewardModel,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001,
                 optimizer_type: str = "adam",
                 loss_function: str = "mse",
                 device: str = 'auto',
                 gradient_clip_norm: Optional[float] = None,
                 use_mixed_precision: bool = False):
        """
        Initialize the reward model trainer.
        
        Args:
            model: The reward model to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay (L2 regularization)
            optimizer_type: Type of optimizer ("adam", "adamw", "sgd", "rmsprop")
            loss_function: Loss function to use ("mse", "mae", "huber", "smooth_l1")
            device: Device to use ('cuda', 'cpu', or 'auto')
            gradient_clip_norm: Maximum norm for gradient clipping (None to disable)
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type.lower()
        self.loss_function = loss_function.lower()
        self.gradient_clip_norm = gradient_clip_norm
        self.use_mixed_precision = use_mixed_precision
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up loss function
        self.loss_fn = self._create_loss_function()
        
        # Set up mixed precision training
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            self.use_mixed_precision = False
            
        # Learning rate scheduler (will be set up in train method)
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_correlation': [],
            'val_correlation': [],
            'learning_rates': [],
            'epochs': 0
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        if self.use_mixed_precision:
            self.logger.info("Mixed precision training enabled")
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        elif self.optimizer_type == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.logger.warning(f"Unknown optimizer type {self.optimizer_type}, using Adam")
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        if self.loss_function == "mse":
            return nn.MSELoss()
        elif self.loss_function == "mae":
            return nn.L1Loss()
        elif self.loss_function == "huber":
            return nn.HuberLoss()
        elif self.loss_function == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            self.logger.warning(f"Unknown loss function {self.loss_function}, using MSE")
            return nn.MSELoss()
            
    def _create_scheduler(self, 
                         scheduler_type: str = "plateau",
                         scheduler_params: Dict[str, Any] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if scheduler_params is None:
            scheduler_params = {}
            
        if scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                verbose=True
            )
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 50),
                eta_min=scheduler_params.get('eta_min', 1e-6)
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_params.get('step_size', 10),
                gamma=scheduler_params.get('gamma', 0.1)
            )
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_params.get('gamma', 0.95)
            )
        else:
            return None
            
    def train_step(self, 
                  states: torch.Tensor, 
                  rewards: torch.Tensor,
                  actions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Perform a single training step with advanced features.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            rewards: Ground truth rewards (batch_size, 1)
            actions: Optional batch of actions (batch_size, action_dim)
            
        Returns:
            Dictionary with training metrics
        """
        # Move data to device
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        
        if actions is not None:
            actions = actions.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                pred_rewards = self.model(states, actions)
                loss = self.loss_fn(pred_rewards, rewards)
                
                # Add ensemble diversity loss if applicable
                if hasattr(self.model, 'get_diversity_loss'):
                    diversity_loss = self.model.get_diversity_loss(states, actions)
                    loss = loss + diversity_loss
        else:
            pred_rewards = self.model(states, actions)
            loss = self.loss_fn(pred_rewards, rewards)
            
            # Add ensemble diversity loss if applicable
            if hasattr(self.model, 'get_diversity_loss'):
                diversity_loss = self.model.get_diversity_loss(states, actions)
                loss = loss + diversity_loss
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
            self.optimizer.step()
        
        # Calculate additional metrics
        with torch.no_grad():
            # Mean Absolute Error
            mae = torch.mean(torch.abs(pred_rewards - rewards)).item()
            
            # Correlation coefficient
            pred_flat = pred_rewards.flatten()
            target_flat = rewards.flatten()
            
            if len(pred_flat) > 1:
                vx = pred_flat - torch.mean(pred_flat)
                vy = target_flat - torch.mean(target_flat)
                correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                correlation = correlation.item()
            else:
                correlation = 0.0
                
            # Accuracy (within threshold)
            threshold = 0.1
            accuracy = torch.mean((torch.abs(pred_rewards - rewards) < threshold).float()).item()
        
        return {
            'loss': loss.item(),
            'mae': mae,
            'correlation': correlation,
            'accuracy': accuracy
        }
        
    def evaluate(self, 
                states: torch.Tensor, 
                rewards: torch.Tensor,
                actions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate model on validation data with comprehensive metrics.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            rewards: Ground truth rewards (batch_size, 1)
            actions: Optional batch of actions (batch_size, action_dim)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Move data to device
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        
        if actions is not None:
            actions = actions.to(self.device)
            
        with torch.no_grad():
            # Forward pass
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    pred_rewards = self.model(states, actions)
                    val_loss = self.loss_fn(pred_rewards, rewards).item()
            else:
                pred_rewards = self.model(states, actions)
                val_loss = self.loss_fn(pred_rewards, rewards).item()
            
            # Additional metrics
            mae = torch.mean(torch.abs(pred_rewards - rewards)).item()
            
            # Correlation coefficient
            pred_flat = pred_rewards.flatten()
            target_flat = rewards.flatten()
            
            if len(pred_flat) > 1:
                vx = pred_flat - torch.mean(pred_flat)
                vy = target_flat - torch.mean(target_flat)
                correlation = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                correlation = correlation.item()
            else:
                correlation = 0.0
                
            # Accuracy (within threshold)
            threshold = 0.1
            accuracy = torch.mean((torch.abs(pred_rewards - rewards) < threshold).float()).item()
            
            # R-squared
            ss_res = torch.sum((pred_rewards - rewards) ** 2).item()
            ss_tot = torch.sum((rewards - torch.mean(rewards)) ** 2).item()
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
        self.model.train()
        
        metrics = {
            'val_loss': val_loss,
            'val_mae': mae,
            'val_correlation': correlation,
            'val_accuracy': accuracy,
            'val_r_squared': r_squared
        }
        
        return metrics
        
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: Optional[torch.utils.data.DataLoader] = None,
             epochs: int = 10,
             early_stopping_patience: int = 5,
             early_stopping_metric: str = "val_loss",
             early_stopping_mode: str = "min",
             scheduler_type: str = "plateau",
             scheduler_params: Dict[str, Any] = None,
             verbose: bool = True,
             save_best: Optional[str] = None,
             save_checkpoint_every: Optional[int] = None,
             resume_from_checkpoint: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs with advanced features.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            early_stopping_metric: Metric to use for early stopping
            early_stopping_mode: "min" or "max" for early stopping
            scheduler_type: Type of learning rate scheduler
            scheduler_params: Parameters for the scheduler
            verbose: Whether to print progress
            save_best: Path to save best model, or None
            save_checkpoint_every: Save checkpoint every N epochs
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from_checkpoint and torch.cuda.is_available():
            start_epoch = self._load_checkpoint(resume_from_checkpoint)
            
        # Set up scheduler
        if scheduler_type != "none":
            self.scheduler = self._create_scheduler(scheduler_type, scheduler_params)
            
        # Early stopping setup
        best_metric = float('inf') if early_stopping_mode == "min" else float('-inf')
        patience_counter = 0
        start_time = time.time()
        
        # Track epochs
        initial_epochs = self.history['epochs']
        self.history['epochs'] += epochs
        
        for epoch in range(start_epoch, start_epoch + epochs):
            epoch_start_time = time.time()
            
            # Training loop
            self.model.train()
            train_metrics = defaultdict(list)
            
            train_pbar = tqdm(train_loader, disable=not verbose, desc=f"Epoch {epoch+1}/{start_epoch+epochs}")
            for batch_idx, batch in enumerate(train_pbar):
                # Extract batch data
                if len(batch) == 2:  # (states, rewards)
                    states, rewards = batch
                    actions = None
                else:  # (states, actions, rewards)
                    states, actions, rewards = batch
                    
                # Training step
                step_metrics = self.train_step(states, rewards, actions)
                
                # Accumulate metrics
                for key, value in step_metrics.items():
                    train_metrics[key].append(value)
                    
                # Update progress bar
                if verbose:
                    train_pbar.set_postfix({
                        'Loss': f"{step_metrics['loss']:.6f}",
                        'Corr': f"{step_metrics['correlation']:.4f}"
                    })
                    
            # Calculate average training metrics
            avg_train_metrics = {key: np.mean(values) for key, values in train_metrics.items()}
            
            # Store training metrics
            self.history['train_loss'].append(avg_train_metrics['loss'])
            self.history['train_correlation'].append(avg_train_metrics['correlation'])
            self.history['train_accuracy'].append(avg_train_metrics['accuracy'])
            
            # Store learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate_on_loader(val_loader)
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_correlation'].append(val_metrics['val_correlation'])
                self.history['val_accuracy'].append(val_metrics['val_accuracy'])
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                        
                # Early stopping check
                current_metric = val_metrics.get(early_stopping_metric, val_metrics['val_loss'])
                
                if early_stopping_mode == "min":
                    is_better = current_metric < best_metric
                else:
                    is_better = current_metric > best_metric
                    
                if is_better:
                    best_metric = current_metric
                    patience_counter = 0
                    
                    # Save best model
                    if save_best is not None:
                        self.model.save(save_best)
                        if verbose:
                            self.logger.info(f"Saved best model to {save_best}")
                            
                    # Store best model state
                    self.best_model_state = self.model.state_dict().copy()
                    self.best_val_loss = val_metrics['val_loss']
                else:
                    patience_counter += 1
                    
            else:
                # No validation data, just step scheduler
                if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                    
            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            if verbose:
                summary = (f"Epoch {epoch+1}/{start_epoch+epochs} "
                          f"({epoch_time:.1f}s) | "
                          f"Train Loss: {avg_train_metrics['loss']:.6f} | "
                          f"Train Corr: {avg_train_metrics['correlation']:.4f}")
                
                if val_metrics:
                    summary += (f" | Val Loss: {val_metrics['val_loss']:.6f} | "
                               f"Val Corr: {val_metrics['val_correlation']:.4f}")
                    
                summary += f" | LR: {current_lr:.2e}"
                
                print(summary)
                
            # Save checkpoint
            if save_checkpoint_every and (epoch + 1) % save_checkpoint_every == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
                self._save_checkpoint(checkpoint_path, epoch + 1)
                if verbose:
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
            # Check early stopping
            if val_loader is not None and patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Report training time
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            if self.best_model_state is not None:
                print(f"Best validation loss: {self.best_val_loss:.6f}")
                
        return self.history
        
    def evaluate_on_loader(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a full data loader with comprehensive metrics.
        
        Args:
            data_loader: DataLoader with validation data
            
        Returns:
            Dictionary with validation metrics
        """
        all_metrics = defaultdict(list)
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                # Extract batch data
                if len(batch) == 2:  # (states, rewards)
                    states, rewards = batch
                    actions = None
                else:  # (states, actions, rewards)
                    states, actions, rewards = batch
                    
                # Evaluate batch
                batch_metrics = self.evaluate(states, rewards, actions)
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
                    
        # Calculate averages
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        return avg_metrics
        
    def _save_checkpoint(self, path: str, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state
        }
        
        torch.save(checkpoint, path)
        
    def _load_checkpoint(self, path: str) -> int:
        """Load training checkpoint and return epoch to resume from."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model_state = checkpoint['best_model_state']
        
        return checkpoint['epoch']
        
    def load_best_model(self) -> None:
        """Load the best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            self.logger.warning("No best model state available")
            
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.history['train_loss']:
            return {"error": "No training history available"}
            
        summary = {
            'total_epochs': len(self.history['train_loss']),
            'best_train_loss': min(self.history['train_loss']),
            'final_train_loss': self.history['train_loss'][-1],
            'best_train_correlation': max(self.history['train_correlation']),
            'final_train_correlation': self.history['train_correlation'][-1],
        }
        
        if self.history['val_loss']:
            summary.update({
                'best_val_loss': min(self.history['val_loss']),
                'final_val_loss': self.history['val_loss'][-1],
                'best_val_correlation': max(self.history['val_correlation']),
                'final_val_correlation': self.history['val_correlation'][-1],
            })
            
        return summary
