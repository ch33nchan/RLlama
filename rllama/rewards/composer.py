#!/usr/bin/env python3
"""
Reward composer for combining multiple reward components.
This module provides sophisticated composition strategies for reward engineering.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Union, Callable
import logging
import time
from collections import defaultdict, deque

from .base import BaseReward

class RewardComposer:
    """
    Advanced composer for multiple reward components with sophisticated combination strategies.
    Supports weighted combination, adaptive weighting, and temporal composition patterns.
    """
    
    def __init__(self, 
                 components: List[BaseReward],
                 composition_strategy: str = "weighted_sum",
                 normalization: str = "none",
                 temporal_window: int = 100,
                 adaptive_weights: bool = False,
                 weight_adaptation_rate: float = 0.01):
        """
        Initialize the reward composer.
        
        Args:
            components: List of reward components to compose
            composition_strategy: Strategy for combining rewards ("weighted_sum", "product", "max", "min", "adaptive")
            normalization: Normalization strategy ("none", "z_score", "min_max", "robust")
            temporal_window: Window size for temporal statistics
            adaptive_weights: Whether to adapt component weights based on performance
            weight_adaptation_rate: Rate of weight adaptation
        """
        self.components = components
        self.composition_strategy = composition_strategy
        self.normalization = normalization
        self.temporal_window = temporal_window
        self.adaptive_weights = adaptive_weights
        self.weight_adaptation_rate = weight_adaptation_rate
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize component tracking
        self.component_names = [comp.__class__.__name__ for comp in components]
        self.component_weights = {name: 1.0 for name in self.component_names}
        
        # Statistics tracking for normalization and adaptation
        self.component_statistics = {
            name: {
                'values': deque(maxlen=temporal_window),
                'mean': 0.0,
                'std': 1.0,
                'min': float('inf'),
                'max': float('-inf'),
                'median': 0.0,
                'mad': 1.0  # Median Absolute Deviation
            }
            for name in self.component_names
        }
        
        # Performance tracking for adaptive weights
        self.performance_history = deque(maxlen=temporal_window)
        self.component_correlations = {name: 0.0 for name in self.component_names}
        
    def calculate(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate rewards from all components and compose them.
        
        Args:
            context: Dictionary containing all needed context for reward calculation
                     
        Returns:
            Dictionary mapping component names to their reward values
        """
        component_rewards = {}
        
        # Calculate individual component rewards
        for component in self.components:
            try:
                component_name = component.__class__.__name__
                reward = component.calculate(context)
                
                # Ensure reward is a float
                if not isinstance(reward, (int, float)):
                    self.logger.warning(f"Component {component_name} returned non-numeric reward: {reward}")
                    reward = 0.0
                else:
                    reward = float(reward)
                
                component_rewards[component_name] = reward
                
                # Update statistics
                self._update_component_statistics(component_name, reward)
                
            except Exception as e:
                self.logger.error(f"Error calculating reward in {component.__class__.__name__}: {e}")
                component_name = component.__class__.__name__
                component_rewards[component_name] = 0.0
                
        # Apply normalization if requested
        if self.normalization != "none":
            component_rewards = self._normalize_rewards(component_rewards)
            
        # Update adaptive weights if enabled
        if self.adaptive_weights:
            self._update_adaptive_weights(component_rewards, context)
            
        # Store original rewards for reference
        original_rewards = component_rewards.copy()
        
        # Apply composition strategy
        composed_reward = self._compose_rewards(component_rewards)
        
        # Add composed reward to the result
        result = original_rewards.copy()
        result['_composed'] = composed_reward
        
        return result
        
    def _update_component_statistics(self, component_name: str, reward: float) -> None:
        """Update running statistics for a component."""
        stats = self.component_statistics[component_name]
        
        # Add to values history
        stats['values'].append(reward)
        
        # Update basic statistics
        if len(stats['values']) > 1:
            values = list(stats['values'])
            stats['mean'] = np.mean(values)
            stats['std'] = np.std(values) + 1e-8  # Add small epsilon
            stats['min'] = min(stats['min'], reward)
            stats['max'] = max(stats['max'], reward)
            
            # Update robust statistics
            stats['median'] = np.median(values)
            mad = np.median(np.abs(values - stats['median']))
            stats['mad'] = max(mad, 1e-8)
        else:
            stats['mean'] = reward
            stats['min'] = reward
            stats['max'] = reward
            stats['median'] = reward
            
    def _normalize_rewards(self, component_rewards: Dict[str, float]) -> Dict[str, float]:
        """Apply normalization to component rewards."""
        normalized_rewards = {}
        
        for name, reward in component_rewards.items():
            stats = self.component_statistics.get(name)
            
            if stats is None or len(stats['values']) < 2:
                normalized_rewards[name] = reward
                continue
                
            if self.normalization == "z_score":
                # Z-score normalization
                normalized = (reward - stats['mean']) / stats['std']
                
            elif self.normalization == "min_max":
                # Min-max normalization
                if stats['max'] > stats['min']:
                    normalized = (reward - stats['min']) / (stats['max'] - stats['min'])
                else:
                    normalized = 0.0
                    
            elif self.normalization == "robust":
                # Robust normalization using median and MAD
                normalized = (reward - stats['median']) / stats['mad']
                
            else:
                normalized = reward
                
            normalized_rewards[name] = normalized
            
        return normalized_rewards
        
    def _compose_rewards(self, component_rewards: Dict[str, float]) -> float:
        """Compose individual rewards into a single value."""
        if not component_rewards:
            return 0.0
            
        rewards = list(component_rewards.values())
        names = list(component_rewards.keys())
        
        if self.composition_strategy == "weighted_sum":
            # Weighted sum of components
            total = 0.0
            total_weight = 0.0
            
            for name, reward in component_rewards.items():
                weight = self.component_weights.get(name, 1.0)
                total += weight * reward
                total_weight += weight
                
            return total / max(total_weight, 1e-8)
            
        elif self.composition_strategy == "product":
            # Product of components (useful for multiplicative rewards)
            product = 1.0
            for reward in rewards:
                # Shift to positive values for product
                product *= (reward + 1.0)
            return product - 1.0
            
        elif self.composition_strategy == "max":
            # Maximum of components
            return max(rewards)
            
        elif self.composition_strategy == "min":
            # Minimum of components
            return min(rewards)
            
        elif self.composition_strategy == "adaptive":
            # Adaptive composition based on component performance
            return self._adaptive_composition(component_rewards)
            
        elif self.composition_strategy == "harmonic_mean":
            # Harmonic mean (useful when all components should be positive)
            positive_rewards = [max(r, 1e-8) for r in rewards]
            harmonic = len(positive_rewards) / sum(1.0 / r for r in positive_rewards)
            return harmonic
            
        elif self.composition_strategy == "geometric_mean":
            # Geometric mean
            if all(r > 0 for r in rewards):
                geometric = np.prod(rewards) ** (1.0 / len(rewards))
                return geometric
            else:
                # Fall back to arithmetic mean for non-positive values
                return np.mean(rewards)
                
        else:
            # Default to simple mean
            return np.mean(rewards)
            
    def _adaptive_composition(self, component_rewards: Dict[str, float]) -> float:
        """Adaptive composition based on component correlations with performance."""
        if not self.performance_history:
            # Fall back to weighted sum if no performance history
            return self._compose_rewards_weighted_sum(component_rewards)
            
        # Weight components by their correlation with past performance
        total = 0.0
        total_weight = 0.0
        
        for name, reward in component_rewards.items():
            # Use correlation as weight (higher correlation = higher weight)
            correlation = abs(self.component_correlations.get(name, 0.0))
            weight = max(correlation, 0.1)  # Minimum weight to avoid zero
            
            total += weight * reward
            total_weight += weight
            
        return total / max(total_weight, 1e-8)
        
    def _compose_rewards_weighted_sum(self, component_rewards: Dict[str, float]) -> float:
        """Helper method for weighted sum composition."""
        total = 0.0
        total_weight = 0.0
        
        for name, reward in component_rewards.items():
            weight = self.component_weights.get(name, 1.0)
            total += weight * reward
            total_weight += weight
            
        return total / max(total_weight, 1e-8)
        
    def _update_adaptive_weights(self, component_rewards: Dict[str, float], context: Dict[str, Any]) -> None:
        """Update component weights based on performance correlation."""
        # Extract performance signal from context
        performance = context.get('performance', context.get('success', context.get('score', 0.0)))
        
        if not isinstance(performance, (int, float)):
            return
            
        # Add to performance history
        self.performance_history.append(performance)
        
        # Need sufficient history for correlation calculation
        if len(self.performance_history) < 10:
            return
            
        # Calculate correlations between each component and performance
        performance_values = list(self.performance_history)
        
        for name in self.component_names:
            stats = self.component_statistics[name]
            
            if len(stats['values']) < 10:
                continue
                
            # Get recent component values (same length as performance history)
            component_values = list(stats['values'])[-len(performance_values):]
            
            if len(component_values) == len(performance_values):
                # Calculate correlation
                correlation = np.corrcoef(component_values, performance_values)[0, 1]
                
                if not np.isnan(correlation):
                    # Update correlation with exponential moving average
                    old_corr = self.component_correlations[name]
                    self.component_correlations[name] = (
                        (1 - self.weight_adaptation_rate) * old_corr +
                        self.weight_adaptation_rate * correlation
                    )
                    
                    # Update component weight based on correlation
                    if self.adaptive_weights:
                        # Positive correlation increases weight, negative decreases
                        weight_adjustment = self.weight_adaptation_rate * correlation
                        old_weight = self.component_weights[name]
                        new_weight = max(0.1, old_weight + weight_adjustment)
                        self.component_weights[name] = new_weight
                        
    def set_component_weight(self, component_name: str, weight: float) -> None:
        """Set the weight for a specific component."""
        if component_name in self.component_weights:
            self.component_weights[component_name] = max(0.0, weight)
        else:
            self.logger.warning(f"Component {component_name} not found")
            
    def get_component_weights(self) -> Dict[str, float]:
        """Get current component weights."""
        return self.component_weights.copy()
        
    def get_component_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get current component statistics."""
        stats_summary = {}
        
        for name, stats in self.component_statistics.items():
            stats_summary[name] = {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'median': stats['median'],
                'mad': stats['mad'],
                'samples': len(stats['values'])
            }
            
        return stats_summary
        
    def get_performance_correlations(self) -> Dict[str, float]:
        """Get correlations between components and performance."""
        return self.component_correlations.copy()
        
    def reset_statistics(self) -> None:
        """Reset all component statistics."""
        for stats in self.component_statistics.values():
            stats['values'].clear()
            stats['mean'] = 0.0
            stats['std'] = 1.0
            stats['min'] = float('inf')
            stats['max'] = float('-inf')
            stats['median'] = 0.0
            stats['mad'] = 1.0
            
        self.performance_history.clear()
        self.component_correlations = {name: 0.0 for name in self.component_names}
        
        self.logger.info("Component statistics reset")
        
    def add_component(self, component: BaseReward, weight: float = 1.0) -> None:
        """Add a new component to the composer."""
        self.components.append(component)
        component_name = component.__class__.__name__
        
        if component_name not in self.component_names:
            self.component_names.append(component_name)
            self.component_weights[component_name] = weight
            self.component_statistics[component_name] = {
                'values': deque(maxlen=self.temporal_window),
                'mean': 0.0,
                'std': 1.0,
                'min': float('inf'),
                'max': float('-inf'),
                'median': 0.0,
                'mad': 1.0
            }
            self.component_correlations[component_name] = 0.0
            
        self.logger.info(f"Added component {component_name} with weight {weight}")
        
    def remove_component(self, component_name: str) -> bool:
        """Remove a component from the composer."""
        # Find and remove the component
        for i, comp in enumerate(self.components):
            if comp.__class__.__name__ == component_name:
                self.components.pop(i)
                break
        else:
            self.logger.warning(f"Component {component_name} not found")
            return False
            
        # Clean up tracking data
        if component_name in self.component_names:
            self.component_names.remove(component_name)
            del self.component_weights[component_name]
            del self.component_statistics[component_name]
            del self.component_correlations[component_name]
            
        self.logger.info(f"Removed component {component_name}")
        return True
