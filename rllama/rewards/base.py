#!/usr/bin/env python3
"""
Base reward class and utilities for the RLlama reward engineering framework.
Provides the foundation for all reward components with advanced features.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple, Callable
import time
import logging
import numpy as np
from collections import defaultdict, deque
import warnings

class RewardMetrics:
    """Container for reward component metrics and statistics."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize reward metrics tracking.
        
        Args:
            window_size: Size of sliding window for statistics
        """
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.call_count = 0
        
        # Running statistics
        self.running_mean = 0.0
        self.running_var = 0.0
        self.running_min = float('inf')
        self.running_max = float('-inf')
        
    def update(self, value: float) -> None:
        """Update metrics with a new value."""
        self.values.append(value)
        self.timestamps.append(time.time())
        self.call_count += 1
        
        # Update running statistics
        if len(self.values) == 1:
            self.running_mean = value
            self.running_var = 0.0
            self.running_min = value
            self.running_max = value
        else:
            # Welford's online algorithm for mean and variance
            n = len(self.values)
            delta = value - self.running_mean
            self.running_mean += delta / n
            delta2 = value - self.running_mean
            self.running_var += delta * delta2
            
            # Update min/max
            self.running_min = min(self.running_min, value)
            self.running_max = max(self.running_max, value)
            
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics."""
        if not self.values:
            return {}
            
        n = len(self.values)
        variance = self.running_var / max(1, n - 1)
        std = np.sqrt(variance)
        
        return {
            'mean': self.running_mean,
            'std': std,
            'min': self.running_min,
            'max': self.running_max,
            'count': self.call_count,
            'recent_count': n
        }

class BaseReward(ABC):
    """
    Abstract base class for all reward components with advanced features.
    
    This class provides:
    - Abstract interface for reward calculation
    - Built-in metrics tracking and statistics
    - Validation and error handling
    - Debugging and profiling capabilities
    - Configuration management
    """

    def __init__(self, 
                 name: Optional[str] = None,
                 weight: float = 1.0,
                 enabled: bool = True,
                 debug: bool = False,
                 track_metrics: bool = True,
                 validate_inputs: bool = True,
                 **kwargs):
        """
        Initialize the base reward component.
        
        Args:
            name: Optional name for the component (defaults to class name)
            weight: Default weight for this component
            enabled: Whether this component is enabled
            debug: Whether to enable debug logging
            track_metrics: Whether to track performance metrics
            validate_inputs: Whether to validate inputs
            **kwargs: Additional parameters specific to subclasses
        """
        # Component identification
        self.name = name or self.__class__.__name__
        self.weight = weight
        self.enabled = enabled
        
        # Configuration
        self.debug = debug
        self.track_metrics = track_metrics
        self.validate_inputs = validate_inputs
        self.config = kwargs
        
        # Set up logging
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
            
        # Metrics tracking
        if track_metrics:
            self.metrics = RewardMetrics()
        else:
            self.metrics = None
            
        # Performance tracking
        self.total_compute_time = 0.0
        self.total_calls = 0
        self.error_count = 0
        self.last_error = None
        
        # Validation tracking
        self.validation_errors = []
        
        # State tracking for debugging
        self.last_context = None
        self.last_result = None
        self.last_computation_time = 0.0
        
        if self.debug:
            self.logger.debug(f"Initialized {self.name} with config: {kwargs}")

    @abstractmethod
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate the reward for a given context.

        This method must be implemented by all subclasses.

        Args:
            context: Dictionary containing all necessary information for 
                    reward calculation (e.g., prompt, response, metadata)

        Returns:
            The calculated reward value for this component
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If context is invalid
        """
        pass
        
    def compute(self, context: Dict[str, Any]) -> float:
        """
        Public interface for reward computation with built-in features.
        
        This method wraps the abstract calculate() method with:
        - Input validation
        - Error handling
        - Performance tracking
        - Metrics collection
        - Debug logging
        
        Args:
            context: Dictionary containing context for reward calculation
            
        Returns:
            The calculated reward value, or 0.0 if disabled or error occurred
        """
        # Check if component is enabled
        if not self.enabled:
            if self.debug:
                self.logger.debug(f"{self.name} is disabled, returning 0.0")
            return 0.0
            
        start_time = time.time()
        
        try:
            # Input validation
            if self.validate_inputs:
                self._validate_context(context)
                
            # Store context for debugging
            if self.debug:
                self.last_context = context.copy()
                
            # Call the actual calculation method
            result = self.calculate(context)
            
            # Validate result
            if not isinstance(result, (int, float)):
                raise ValueError(f"calculate() must return a number, got {type(result)}")
                
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"calculate() returned invalid value: {result}")
                
            result = float(result)
            
            # Track metrics
            if self.metrics:
                self.metrics.update(result)
                
            # Store result for debugging
            if self.debug:
                self.last_result = result
                self.logger.debug(f"{self.name} computed reward: {result}")
                
            return result
            
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            
            self.logger.error(f"Error in {self.name}: {e}")
            
            if self.debug:
                self.logger.debug(f"Context that caused error: {context}")
                
            # Return 0.0 on error to avoid breaking the pipeline
            return 0.0
            
        finally:
            # Track performance
            computation_time = time.time() - start_time
            self.total_compute_time += computation_time
            self.total_calls += 1
            self.last_computation_time = computation_time
            
            if self.debug:
                self.logger.debug(f"{self.name} computation took {computation_time:.6f}s")
                
    def _validate_context(self, context: Dict[str, Any]) -> None:
        """
        Validate the input context.
        
        Args:
            context: Context dictionary to validate
            
        Raises:
            ValueError: If context is invalid
        """
        if not isinstance(context, dict):
            raise ValueError(f"Context must be a dictionary, got {type(context)}")
            
        if not context:
            warnings.warn(f"{self.name}: Empty context provided")
            
        # Subclasses can override this for specific validation
        self._validate_specific_context(context)
        
    def _validate_specific_context(self, context: Dict[str, Any]) -> None:
        """
        Validate context specific to this reward component.
        
        Subclasses should override this method to implement specific validation.
        
        Args:
            context: Context dictionary to validate
        """
        pass
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for this component.
        
        Returns:
            Dictionary containing performance and statistical metrics
        """
        base_metrics = {
            'name': self.name,
            'enabled': self.enabled,
            'total_calls': self.total_calls,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.total_calls),
            'total_compute_time': self.total_compute_time,
            'avg_compute_time': self.total_compute_time / max(1, self.total_calls),
            'last_computation_time': self.last_computation_time,
            'last_error': self.last_error
        }
        
        # Add statistical metrics if available
        if self.metrics:
            stats = self.metrics.get_statistics()
            base_metrics.update(stats)
            
        return base_metrics
        
    def reset_metrics(self) -> None:
        """Reset all metrics and statistics."""
        if self.metrics:
            self.metrics = RewardMetrics()
            
        self.total_compute_time = 0.0
        self.total_calls = 0
        self.error_count = 0
        self.last_error = None
        self.validation_errors = []
        
        if self.debug:
            self.logger.debug(f"Reset metrics for {self.name}")
            
    def enable(self) -> None:
        """Enable this reward component."""
        self.enabled = True
        if self.debug:
            self.logger.debug(f"Enabled {self.name}")
            
    def disable(self) -> None:
        """Disable this reward component."""
        self.enabled = False
        if self.debug:
            self.logger.debug(f"Disabled {self.name}")
            
    def set_weight(self, weight: float) -> None:
        """
        Set the weight for this component.
        
        Args:
            weight: New weight value
        """
        old_weight = self.weight
        self.weight = float(weight)
        
        if self.debug:
            self.logger.debug(f"Changed weight for {self.name}: {old_weight} -> {self.weight}")
            
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of this component.
        
        Returns:
            Dictionary containing component configuration
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'weight': self.weight,
            'enabled': self.enabled,
            'debug': self.debug,
            'track_metrics': self.track_metrics,
            'validate_inputs': self.validate_inputs,
            **self.config
        }
        
    def update_config(self, **kwargs) -> None:
        """
        Update component configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if self.debug:
                    self.logger.debug(f"Updated {self.name}.{key} = {value}")
            else:
                self.config[key] = value
                if self.debug:
                    self.logger.debug(f"Added config {self.name}.{key} = {value}")
                    
    def __str__(self) -> str:
        """String representation of the component."""
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name}(weight={self.weight}, {status})"
        
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"weight={self.weight}, enabled={self.enabled})")
                
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get detailed debug information.
        
        Returns:
            Dictionary with comprehensive debug information
        """
        return {
            'component_info': {
                'name': self.name,
                'class': self.__class__.__name__,
                'module': self.__class__.__module__,
                'enabled': self.enabled,
                'weight': self.weight
            },
            'performance': {
                'total_calls': self.total_calls,
                'total_compute_time': self.total_compute_time,
                'avg_compute_time': self.total_compute_time / max(1, self.total_calls),
                'last_computation_time': self.last_computation_time
            },
            'errors': {
                'error_count': self.error_count,
                'error_rate': self.error_count / max(1, self.total_calls),
                'last_error': self.last_error,
                'validation_errors': self.validation_errors[-10:]  # Last 10 errors
            },
            'last_execution': {
                'context': self.last_context,
                'result': self.last_result,
                'computation_time': self.last_computation_time
            },
            'configuration': self.get_config(),
            'metrics': self.get_metrics() if self.metrics else None
        }

class CompositeReward(BaseReward):
    """
    Base class for reward components that combine multiple sub-components.
    Useful for creating hierarchical reward structures.
    """
    
    def __init__(self, 
                 components: List[BaseReward],
                 combination_method: str = "weighted_sum",
                 **kwargs):
        """
        Initialize composite reward.
        
        Args:
            components: List of sub-components
            combination_method: How to combine sub-rewards
            **kwargs: Additional arguments for BaseReward
        """
        super().__init__(**kwargs)
        self.components = components
        self.combination_method = combination_method
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate reward by combining sub-components."""
        if not self.components:
            return 0.0
            
        # Calculate rewards from all components
        rewards = []
        weights = []
        
        for component in self.components:
            if component.enabled:
                reward = component.compute(context)
                rewards.append(reward)
                weights.append(component.weight)
                
        if not rewards:
            return 0.0
            
        # Combine based on method
        if self.combination_method == "weighted_sum":
            return sum(w * r for w, r in zip(weights, rewards))
        elif self.combination_method == "mean":
            return np.mean(rewards)
        elif self.combination_method == "max":
            return max(rewards)
        elif self.combination_method == "min":
            return min(rewards)
        else:
            # Default to weighted sum
            return sum(w * r for w, r in zip(weights, rewards))
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including sub-component metrics."""
        metrics = super().get_metrics()
        metrics['sub_components'] = [comp.get_metrics() for comp in self.components]
        return metrics

class CachedReward(BaseReward):
    """
    Base class for reward components that cache results to avoid recomputation.
    Useful for expensive reward calculations.
    """
    
    def __init__(self, 
                 cache_size: int = 1000,
                 cache_key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
                 **kwargs):
        """
        Initialize cached reward.
        
        Args:
            cache_size: Maximum number of cached results
            cache_key_fn: Function to generate cache keys from context
            **kwargs: Additional arguments for BaseReward
        """
        super().__init__(**kwargs)
        self.cache_size = cache_size
        self.cache_key_fn = cache_key_fn or self._default_cache_key
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _default_cache_key(self, context: Dict[str, Any]) -> str:
        """Default cache key generation."""
        # Simple hash of string representation
        # Subclasses should override for better cache keys
        return str(hash(str(sorted(context.items()))))
        
    def compute(self, context: Dict[str, Any]) -> float:
        """Compute with caching."""
        if not self.enabled:
            return 0.0
            
        # Generate cache key
        cache_key = self.cache_key_fn(context)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            if self.debug:
                self.logger.debug(f"{self.name} cache hit for key: {cache_key}")
            return self.cache[cache_key]
            
        # Cache miss - compute normally
        self.cache_misses += 1
        result = super().compute(context)
        
        # Store in cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[cache_key] = result
        
        if self.debug:
            self.logger.debug(f"{self.name} cache miss, computed and cached: {cache_key}")
            
        return result
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics including cache statistics."""
        metrics = super().get_metrics()
        total_requests = self.cache_hits + self.cache_misses
        
        metrics.update({
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, total_requests),
            'cache_size': len(self.cache),
            'cache_capacity': self.cache_size
        })
        
        return metrics
        
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.debug:
            self.logger.debug(f"Cleared cache for {self.name}")
