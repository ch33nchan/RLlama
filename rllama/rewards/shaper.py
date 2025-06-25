#!/usr/bin/env python3
"""
Advanced reward shaping module for RLlama.
Provides sophisticated reward scheduling and composition strategies.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
import math
from enum import Enum
import logging

class ScheduleType(Enum):
    """Types of reward schedule functions"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"
    POLYNOMIAL = "polynomial"
    WARMUP_COSINE = "warmup_cosine"
    CYCLIC = "cyclic"
    CUSTOM = "custom"

class RewardConfig:
    """Configuration for a reward component with advanced scheduling."""
    
    def __init__(self, 
                 name: str,
                 weight: float = 1.0,
                 schedule_type: Union[ScheduleType, str] = ScheduleType.CONSTANT,
                 schedule_params: Dict[str, Any] = None,
                 min_weight: float = 0.0,
                 max_weight: float = float('inf'),
                 custom_schedule_fn: Optional[Callable[[int, Dict[str, Any]], float]] = None,
                 adaptive: bool = False,
                 adaptation_rate: float = 0.01):
        """
        Initialize a reward configuration.
        
        Args:
            name: Name of the reward component
            weight: Initial/base weight
            schedule_type: Type of weight schedule
            schedule_params: Parameters for the schedule
            min_weight: Minimum weight value
            max_weight: Maximum weight value
            custom_schedule_fn: Custom schedule function if schedule_type is CUSTOM
            adaptive: Whether to use adaptive weight adjustment
            adaptation_rate: Rate of adaptation for adaptive weights
        """
        self.name = name
        self.base_weight = weight
        self.current_weight = weight
        
        # Convert string to enum if needed
        if isinstance(schedule_type, str):
            try:
                self.schedule_type = ScheduleType(schedule_type.lower())
            except ValueError:
                self.schedule_type = ScheduleType.CONSTANT
        else:
            self.schedule_type = schedule_type
            
        self.schedule_params = schedule_params or {}
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.custom_schedule_fn = custom_schedule_fn
        self.adaptive = adaptive
        self.adaptation_rate = adaptation_rate
        
        # Adaptive weight tracking
        self.performance_history = []
        self.weight_history = []
        
    def get_weight(self, step: int, performance: Optional[float] = None) -> float:
        """
        Calculate the weight for a given step based on the schedule.
        
        Args:
            step: Current training step
            performance: Optional performance metric for adaptive adjustment
            
        Returns:
            Weight value for the current step
        """
        # Calculate scheduled weight
        if self.schedule_type == ScheduleType.CONSTANT:
            scheduled_weight = self.base_weight
            
        elif self.schedule_type == ScheduleType.LINEAR:
            scheduled_weight = self._linear_schedule(step)
            
        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            scheduled_weight = self._exponential_schedule(step)
            
        elif self.schedule_type == ScheduleType.COSINE:
            scheduled_weight = self._cosine_schedule(step)
            
        elif self.schedule_type == ScheduleType.STEP:
            scheduled_weight = self._step_schedule(step)
            
        elif self.schedule_type == ScheduleType.POLYNOMIAL:
            scheduled_weight = self._polynomial_schedule(step)
            
        elif self.schedule_type == ScheduleType.WARMUP_COSINE:
            scheduled_weight = self._warmup_cosine_schedule(step)
            
        elif self.schedule_type == ScheduleType.CYCLIC:
            scheduled_weight = self._cyclic_schedule(step)
            
        elif self.schedule_type == ScheduleType.CUSTOM and self.custom_schedule_fn:
            scheduled_weight = self.custom_schedule_fn(step, self.schedule_params)
            
        else:
            scheduled_weight = self.base_weight
            
        # Apply adaptive adjustment if enabled
        if self.adaptive and performance is not None:
            scheduled_weight = self._apply_adaptive_adjustment(scheduled_weight, performance)
            
        # Clamp weight to allowed range
        final_weight = max(self.min_weight, min(self.max_weight, scheduled_weight))
        
        # Update current weight and history
        self.current_weight = final_weight
        self.weight_history.append(final_weight)
        if len(self.weight_history) > 1000:  # Keep history bounded
            self.weight_history = self.weight_history[-1000:]
            
        return final_weight
        
    def _linear_schedule(self, step: int) -> float:
        """Linear weight schedule."""
        start_step = self.schedule_params.get("start_step", 0)
        end_step = self.schedule_params.get("end_step", 10000)
        start_value = self.schedule_params.get("start_value", self.base_weight)
        end_value = self.schedule_params.get("end_value", self.base_weight)
        
        if step <= start_step:
            return start_value
        elif step >= end_step:
            return end_value
        else:
            progress = (step - start_step) / (end_step - start_step)
            return start_value + progress * (end_value - start_value)
            
    def _exponential_schedule(self, step: int) -> float:
        """Exponential decay schedule."""
        decay_rate = self.schedule_params.get("decay_rate", 0.9999)
        start_step = self.schedule_params.get("start_step", 0)
        
        if step <= start_step:
            return self.base_weight
        else:
            decay_steps = step - start_step
            return self.base_weight * (decay_rate ** decay_steps)
            
    def _cosine_schedule(self, step: int) -> float:
        """Cosine annealing schedule."""
        start_step = self.schedule_params.get("start_step", 0)
        end_step = self.schedule_params.get("end_step", 10000)
        start_value = self.schedule_params.get("start_value", self.base_weight)
        end_value = self.schedule_params.get("end_value", 0.0)
        
        if step <= start_step:
            return start_value
        elif step >= end_step:
            return end_value
        else:
            progress = (step - start_step) / (end_step - start_step)
            cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
            return end_value + (start_value - end_value) * cosine_term
            
    def _step_schedule(self, step: int) -> float:
        """Step-wise schedule."""
        boundaries = self.schedule_params.get("boundaries", [1000, 2000, 3000])
        values = self.schedule_params.get("values", [self.base_weight, self.base_weight * 0.5, self.base_weight * 0.1, self.base_weight * 0.01])
        
        if len(values) != len(boundaries) + 1:
            raise ValueError("Step schedule needs len(values) == len(boundaries) + 1")
            
        weight = values[0]
        for i, boundary in enumerate(boundaries):
            if step >= boundary:
                weight = values[i + 1]
        return weight
        
    def _polynomial_schedule(self, step: int) -> float:
        """Polynomial decay schedule."""
        start_step = self.schedule_params.get("start_step", 0)
        end_step = self.schedule_params.get("end_step", 10000)
        power = self.schedule_params.get("power", 2.0)
        start_value = self.schedule_params.get("start_value", self.base_weight)
        end_value = self.schedule_params.get("end_value", 0.0)
        
        if step <= start_step:
            return start_value
        elif step >= end_step:
            return end_value
        else:
            progress = (step - start_step) / (end_step - start_step)
            decay_factor = (1 - progress) ** power
            return end_value + (start_value - end_value) * decay_factor
            
    def _warmup_cosine_schedule(self, step: int) -> float:
        """Warmup followed by cosine annealing."""
        warmup_steps = self.schedule_params.get("warmup_steps", 1000)
        total_steps = self.schedule_params.get("total_steps", 10000)
        warmup_start_value = self.schedule_params.get("warmup_start_value", 0.0)
        max_value = self.schedule_params.get("max_value", self.base_weight)
        min_value = self.schedule_params.get("min_value", 0.0)
        
        if step < warmup_steps:
            # Linear warmup
            progress = step / warmup_steps
            return warmup_start_value + progress * (max_value - warmup_start_value)
        else:
            # Cosine annealing
            cosine_steps = step - warmup_steps
            cosine_total = total_steps - warmup_steps
            if cosine_total <= 0:
                return max_value
            progress = cosine_steps / cosine_total
            cosine_term = 0.5 * (1 + math.cos(math.pi * progress))
            return min_value + (max_value - min_value) * cosine_term
            
    def _cyclic_schedule(self, step: int) -> float:
        """Cyclic learning rate schedule."""
        cycle_length = self.schedule_params.get("cycle_length", 2000)
        min_value = self.schedule_params.get("min_value", self.base_weight * 0.1)
        max_value = self.schedule_params.get("max_value", self.base_weight)
        
        cycle_position = (step % cycle_length) / cycle_length
        if cycle_position < 0.5:
            # Increasing phase
            progress = cycle_position * 2
            return min_value + progress * (max_value - min_value)
        else:
            # Decreasing phase
            progress = (cycle_position - 0.5) * 2
            return max_value - progress * (max_value - min_value)
            
    def _apply_adaptive_adjustment(self, scheduled_weight: float, performance: float) -> float:
        """Apply adaptive weight adjustment based on performance."""
        # Add performance to history
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:  # Keep bounded
            self.performance_history = self.performance_history[-100:]
            
        # Need sufficient history for adaptation
        if len(self.performance_history) < 10:
            return scheduled_weight
            
        # Calculate performance trend
        recent_performance = self.performance_history[-5:]
        older_performance = self.performance_history[-10:-5]
        
        recent_avg = np.mean(recent_performance)
        older_avg = np.mean(older_performance)
        
        # Adjust weight based on performance trend
        performance_trend = recent_avg - older_avg
        
        # If performance is improving, maintain or slightly increase weight
        # If performance is declining, adjust weight
        if performance_trend > 0:
            adjustment = 1.0 + self.adaptation_rate * performance_trend
        else:
            adjustment = 1.0 + self.adaptation_rate * performance_trend * 0.5  # More conservative decrease
            
        return scheduled_weight * adjustment

class RewardShaper:
    """
    Advanced reward shaper with sophisticated scheduling and composition strategies.
    Applies weights from a configuration to component rewards to produce final scalar rewards.
    """
    
    def __init__(self, shaping_config: Dict[str, Any]):
        """
        Initialize the reward shaper.
        
        Args:
            shaping_config: The 'shaping_config' block from configuration
        """
        self.config = shaping_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Parse configurations
        self.reward_configs = {}
        self._parse_configurations(shaping_config)
        
        # Global shaping parameters
        self.global_scaling = shaping_config.get("global_scaling", 1.0)
        self.composition_strategy = shaping_config.get("composition_strategy", "weighted_sum")
        self.normalization = shaping_config.get("normalization", "none")
        
        # Statistics for normalization
        self.component_stats = {}
        self.global_stats = {"mean": 0.0, "std": 1.0, "count": 0}
        
    def _parse_configurations(self, shaping_config: Dict[str, Any]) -> None:
        """Parse reward configurations from config dictionary."""
        for name, config in shaping_config.items():
            # Skip global parameters
            if name in ["global_scaling", "composition_strategy", "normalization"]:
                continue
                
            if isinstance(config, dict):
                # Full configuration
                weight = config.get("weight", 1.0)
                schedule_type = config.get("schedule", "constant")
                schedule_params = config.get("schedule_params", {})
                min_weight = config.get("min_weight", 0.0)
                max_weight = config.get("max_weight", float('inf'))
                adaptive = config.get("adaptive", False)
                adaptation_rate = config.get("adaptation_rate", 0.01)
                
                self.reward_configs[name] = RewardConfig(
                    name=name,
                    weight=weight,
                    schedule_type=schedule_type,
                    schedule_params=schedule_params,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    adaptive=adaptive,
                    adaptation_rate=adaptation_rate
                )
            else:
                # Simple case: just a weight value
                self.reward_configs[name] = RewardConfig(
                    name=name,
                    weight=float(config)
                )
                
    def shape(self, 
             component_rewards: Dict[str, float], 
             step: int = 0,
             performance: Optional[float] = None) -> float:
        """
        Shape the final reward by applying weights and composition strategy.

        Args:
            component_rewards: Dictionary of raw reward values from components
            step: Current training step for scheduling
            performance: Optional performance metric for adaptive adjustment

        Returns:
            The final shaped reward as a single float
        """
        if not component_rewards:
            return 0.0
            
        # Update component statistics for normalization
        self._update_component_statistics(component_rewards)
        
        # Apply normalization if requested
        if self.normalization != "none":
            component_rewards = self._normalize_rewards(component_rewards)
            
        # Apply composition strategy
        if self.composition_strategy == "weighted_sum":
            final_reward = self._weighted_sum_composition(component_rewards, step, performance)
        elif self.composition_strategy == "weighted_product":
            final_reward = self._weighted_product_composition(component_rewards, step, performance)
        elif self.composition_strategy == "max":
            final_reward = self._max_composition(component_rewards, step, performance)
        elif self.composition_strategy == "min":
            final_reward = self._min_composition(component_rewards, step, performance)
        elif self.composition_strategy == "harmonic_mean":
            final_reward = self._harmonic_mean_composition(component_rewards, step, performance)
        else:
            # Default to weighted sum
            final_reward = self._weighted_sum_composition(component_rewards, step, performance)
            
        # Apply global scaling
        final_reward *= self.global_scaling
        
        # Update global statistics
        self._update_global_statistics(final_reward)
        
        return final_reward
        
    def _weighted_sum_composition(self, 
                                component_rewards: Dict[str, float], 
                                step: int,
                                performance: Optional[float]) -> float:
        """Weighted sum composition strategy."""
        final_reward = 0.0
        
        for name, reward_val in component_rewards.items():
            if name in self.reward_configs:
                weight = self.reward_configs[name].get_weight(step, performance)
            else:
                weight = 1.0  # Default weight
                
            final_reward += weight * reward_val
            
        return final_reward
        
    def _weighted_product_composition(self, 
                                    component_rewards: Dict[str, float], 
                                    step: int,
                                    performance: Optional[float]) -> float:
        """Weighted product composition strategy."""
        final_reward = 1.0
        
        for name, reward_val in component_rewards.items():
            if name in self.reward_configs:
                weight = self.reward_configs[name].get_weight(step, performance)
            else:
                weight = 1.0
                
            # Shift to positive values and apply weight as exponent
            shifted_reward = reward_val + 1.0
            final_reward *= (shifted_reward ** weight)
            
        return final_reward - 1.0  # Shift back
        
    def _max_composition(self, 
                        component_rewards: Dict[str, float], 
                        step: int,
                        performance: Optional[float]) -> float:
        """Maximum composition strategy with weights as selection probabilities."""
        weighted_rewards = []
        weights = []
        
        for name, reward_val in component_rewards.items():
            if name in self.reward_configs:
                weight = self.reward_configs[name].get_weight(step, performance)
            else:
                weight = 1.0
                
            weighted_rewards.append(reward_val * weight)
            weights.append(weight)
            
        return max(weighted_rewards) if weighted_rewards else 0.0
        
    def _min_composition(self, 
                        component_rewards: Dict[str, float], 
                        step: int,
                        performance: Optional[float]) -> float:
        """Minimum composition strategy with weights."""
        weighted_rewards = []
        
        for name, reward_val in component_rewards.items():
            if name in self.reward_configs:
                weight = self.reward_configs[name].get_weight(step, performance)
            else:
                weight = 1.0
                
            weighted_rewards.append(reward_val * weight)
            
        return min(weighted_rewards) if weighted_rewards else 0.0
        
    def _harmonic_mean_composition(self, 
                                 component_rewards: Dict[str, float], 
                                 step: int,
                                 performance: Optional[float]) -> float:
        """Harmonic mean composition strategy."""
        weighted_rewards = []
        total_weight = 0.0
        
        for name, reward_val in component_rewards.items():
            if name in self.reward_configs:
                weight = self.reward_configs[name].get_weight(step, performance)
            else:
                weight = 1.0
                
            # Ensure positive values for harmonic mean
            positive_reward = max(reward_val, 1e-8)
            weighted_rewards.append(weight / positive_reward)
            total_weight += weight
            
        if not weighted_rewards or total_weight == 0:
            return 0.0
            
        harmonic_sum = sum(weighted_rewards)
        return total_weight / harmonic_sum
        
    def _update_component_statistics(self, component_rewards: Dict[str, float]) -> None:
        """Update statistics for each component."""
        for name, reward in component_rewards.items():
            if name not in self.component_stats:
                self.component_stats[name] = {
                    "values": [],
                    "mean": 0.0,
                    "std": 1.0,
                    "min": reward,
                    "max": reward
                }
                
            stats = self.component_stats[name]
            stats["values"].append(reward)
            
            # Keep only recent values
            if len(stats["values"]) > 1000:
                stats["values"] = stats["values"][-1000:]
                
            # Update statistics
            if len(stats["values"]) > 1:
                stats["mean"] = np.mean(stats["values"])
                stats["std"] = max(np.std(stats["values"]), 1e-8)
                stats["min"] = min(stats["min"], reward)
                stats["max"] = max(stats["max"], reward)
                
    def _normalize_rewards(self, component_rewards: Dict[str, float]) -> Dict[str, float]:
        """Apply normalization to component rewards."""
        normalized = {}
        
        for name, reward in component_rewards.items():
            if name not in self.component_stats:
                normalized[name] = reward
                continue
                
            stats = self.component_stats[name]
            
            if self.normalization == "z_score":
                normalized[name] = (reward - stats["mean"]) / stats["std"]
            elif self.normalization == "min_max":
                if stats["max"] > stats["min"]:
                    normalized[name] = (reward - stats["min"]) / (stats["max"] - stats["min"])
                else:
                    normalized[name] = 0.0
            else:
                normalized[name] = reward
                
        return normalized
        
    def _update_global_statistics(self, final_reward: float) -> None:
        """Update global reward statistics."""
        self.global_stats["count"] += 1
        
        # Running average
        alpha = 1.0 / self.global_stats["count"]
        old_mean = self.global_stats["mean"]
        self.global_stats["mean"] += alpha * (final_reward - old_mean)
        
        # Running standard deviation (simplified)
        self.global_stats["std"] = 0.9 * self.global_stats["std"] + 0.1 * abs(final_reward - old_mean)
        
    def get_current_weights(self, step: int = 0, performance: Optional[float] = None) -> Dict[str, float]:
        """
        Get the current weights for all components at a specific step.
        
        Args:
            step: The current training step
            performance: Optional performance metric
            
        Returns:
            Dictionary mapping component names to their current weights
        """
        return {
            name: config.get_weight(step, performance) 
            for name, config in self.reward_configs.items()
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about reward shaping."""
        return {
            "component_stats": self.component_stats,
            "global_stats": self.global_stats,
            "current_weights": {name: config.current_weight for name, config in self.reward_configs.items()},
            "composition_strategy": self.composition_strategy,
            "normalization": self.normalization,
            "global_scaling": self.global_scaling
        }
        
    def add_component(self, name: str, config: Union[float, Dict[str, Any], RewardConfig]) -> None:
        """
        Add a new reward component configuration.
        
        Args:
            name: Name of the component
            config: Weight value, configuration dict, or RewardConfig object
        """
        if isinstance(config, RewardConfig):
            self.reward_configs[name] = config
        elif isinstance(config, dict):
            self.reward_configs[name] = RewardConfig(
                name=name,
                weight=config.get("weight", 1.0),
                schedule_type=config.get("schedule", "constant"),
                schedule_params=config.get("schedule_params", {}),
                min_weight=config.get("min_weight", 0.0),
                max_weight=config.get("max_weight", float('inf')),
                adaptive=config.get("adaptive", False),
                adaptation_rate=config.get("adaptation_rate", 0.01)
            )
        else:
            self.reward_configs[name] = RewardConfig(
                name=name,
                weight=float(config)
            )
            
    def remove_component(self, name: str) -> bool:
        """
        Remove a component configuration.
        
        Args:
            name: Name of the component to remove
            
        Returns:
            True if component was removed, False if not found
        """
        if name in self.reward_configs:
            del self.reward_configs[name]
            return True
        return False
        
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.component_stats = {}
        self.global_stats = {"mean": 0.0, "std": 1.0, "count": 0}
        
        # Reset component histories
        for config in self.reward_configs.values():
            config.performance_history = []
            config.weight_history = []
