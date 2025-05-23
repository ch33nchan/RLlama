import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import deque

class RewardNormalizer:
    def __init__(self, method: str = "standard", **kwargs):
        self.method = method
        self.params = kwargs
        self.stats = {}
        self.initialized = False
    
    def normalize(self, reward: float, component_name: str = "default") -> float:
        if component_name not in self.stats:
            self.stats[component_name] = {
                "count": 0,
                "mean": 0.0,
                "M2": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "recent_values": deque(maxlen=self.params.get("window_size", 100))
            }
        
        stats = self.stats[component_name]
        stats["count"] += 1
        stats["recent_values"].append(reward)
        
        if reward < stats["min"]:
            stats["min"] = reward
        if reward > stats["max"]:
            stats["max"] = reward
        
        delta = reward - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = reward - stats["mean"]
        stats["M2"] += delta * delta2
        
        if stats["count"] > 1:
            stats["var"] = stats["M2"] / (stats["count"] - 1)
            stats["std"] = np.sqrt(stats["var"])
        else:
            stats["var"] = 0.0
            stats["std"] = 0.0
        
        if self.method == "standard":
            if stats["std"] > 0:
                return (reward - stats["mean"]) / stats["std"]
            else:
                return 0.0
        
        elif self.method == "minmax":
            if stats["max"] > stats["min"]:
                return (reward - stats["min"]) / (stats["max"] - stats["min"])
            else:
                return 0.0
        
        elif self.method == "percentile":
            if len(stats["recent_values"]) > 1:
                sorted_values = sorted(stats["recent_values"])
                rank = sorted_values.index(reward) if reward in sorted_values else 0
                return rank / (len(sorted_values) - 1)
            else:
                return 0.0
        
        elif self.method == "robust":
            q1 = np.percentile(list(stats["recent_values"]), 25)
            q3 = np.percentile(list(stats["recent_values"]), 75)
            iqr = q3 - q1
            if iqr > 0:
                return (reward - stats["mean"]) / iqr
            else:
                return 0.0
        
        elif self.method == "tanh":
            scale = self.params.get("scale", 1.0)
            return np.tanh(reward / scale)
        
        elif self.method == "clip":
            min_val = self.params.get("min", -1.0)
            max_val = self.params.get("max", 1.0)
            return max(min_val, min(max_val, reward))
        
        else:
            return reward
    
    def reset(self, component_name: Optional[str] = None):
        if component_name:
            if component_name in self.stats:
                del self.stats[component_name]
        else:
            self.stats = {}

class PopArtNormalizer(RewardNormalizer):
    def __init__(self, beta: float = 0.0001, epsilon: float = 1e-8):
        super().__init__(method="popart")
        self.beta = beta
        self.epsilon = epsilon
        self.mean = 0.0
        self.std = 1.0
        self.initialized = False
    
    def normalize(self, reward: float, component_name: str = "default") -> float:
        if not self.initialized:
            self.mean = reward
            self.std = 1.0
            self.initialized = True
            return 0.0
        
        normalized_reward = (reward - self.mean) / (self.std + self.epsilon)
        
        self.update(reward)
        
        return normalized_reward
    
    def update(self, reward: float):
        old_mean = self.mean
        old_std = self.std
        
        self.mean = (1 - self.beta) * self.mean + self.beta * reward
        self.std = np.sqrt((1 - self.beta) * (self.std**2 + (old_mean - self.mean)**2) + 
                          self.beta * (reward - self.mean)**2)
        self.std = max(self.std, self.epsilon)
    
    def denormalize(self, normalized_reward: float) -> float:
        return normalized_reward * self.std + self.mean
    
    def reset(self, component_name: Optional[str] = None):
        self.mean = 0.0
        self.std = 1.0
        self.initialized = False

class AdaptiveNormalizer(RewardNormalizer):
    def __init__(self, 
                 initial_method: str = "standard", 
                 phase_thresholds: Dict[str, int] = None,
                 **kwargs):
        super().__init__(method=initial_method, **kwargs)
        self.initial_method = initial_method
        self.phase_thresholds = phase_thresholds or {
            "exploration": 1000,
            "learning": 10000,
            "exploitation": 50000
        }
        self.step_counter = 0
        self.current_phase = "exploration"
        self.normalizers = {
            "exploration": RewardNormalizer(method="robust", window_size=100),
            "learning": RewardNormalizer(method="standard"),
            "exploitation": PopArtNormalizer(beta=0.0001)
        }
    
    def normalize(self, reward: float, component_name: str = "default") -> float:
        self.step_counter += 1
        self._update_phase()
        
        return self.normalizers[self.current_phase].normalize(reward, component_name)
    
    def _update_phase(self):
        if self.step_counter > self.phase_thresholds.get("exploitation", 50000):
            self.current_phase = "exploitation"
        elif self.step_counter > self.phase_thresholds.get("learning", 10000):
            self.current_phase = "learning"
        else:
            self.current_phase = "exploration"
    
    def reset(self, component_name: Optional[str] = None):
        for normalizer in self.normalizers.values():
            normalizer.reset(component_name)
        self.step_counter = 0
        self.current_phase = "exploration"

import numpy as np
from collections import deque
import logging
from typing import List, Union, Dict # Ensure Dict is imported

logger = logging.getLogger(__name__)

class Normalizer:
    """
    Handles normalization of reward values using various strategies.
    """
    def __init__(self, strategy: str = 'none', window_size: int = 100, epsilon: float = 1e-8):
        """
        Initializes the Normalizer.

        Args:
            strategy (str): Normalization strategy ('none', 'mean_std', 'min_max', 'running_mean_std').
            window_size (int): Window size for running statistics or min/max.
            epsilon (float): Small value to prevent division by zero.
        """
        self.strategy = strategy.lower()
        self.window_size = window_size
        self.epsilon = epsilon
        
        self.values_history = deque(maxlen=self.window_size)
        self.mean = 0.0
        self.std = 1.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.count = 0

        if self.strategy not in ['none', 'mean_std', 'min_max', 'running_mean_std']:
            logger.warning(f"Unknown normalization strategy '{strategy}'. Defaulting to 'none'.")
            self.strategy = 'none'

        logger.debug(f"Normalizer initialized with strategy: {self.strategy}, window: {self.window_size}, epsilon: {self.epsilon}")

    def update(self, values: Union[List[float], np.ndarray, float]):
        """
        Updates the normalizer's statistics with new reward values.

        Args:
            values (Union[List[float], np.ndarray, float]): A single value or a list/array of values.
        """
        if self.strategy == 'none':
            return

        if isinstance(values, (float, int)):
            values = [float(values)]
        
        for value in values:
            self.values_history.append(float(value))
            self.count += 1

            if self.strategy == 'running_mean_std':
                # Welford's algorithm for running mean and variance
                old_mean = self.mean
                self.mean += (value - self.mean) / self.count if self.count > 0 else 0
                if self.count > 1:
                    # M2 is sum of squares of differences from the current mean
                    # For simplicity, we'll recompute std from history if using running_mean_std with a window
                    # A true Welford would update M2. For windowed, it's easier to recompute.
                    pass # Recomputation happens in normalize_value if needed from history

            elif self.strategy == 'min_max':
                self.min_val = min(self.min_val, value)
                self.max_val = max(self.max_val, value)

    def normalize_value(self, value: float) -> float:
        """
        Normalizes a single reward value based on the current statistics.

        Args:
            value (float): The reward value to normalize.

        Returns:
            float: The normalized reward value.
        """
        if self.strategy == 'none' or not self.values_history:
            return value

        current_values = np.array(list(self.values_history))

        if self.strategy == 'mean_std' or self.strategy == 'running_mean_std':
            # For 'mean_std', always use the current window's stats
            # For 'running_mean_std', if using a window, stats are from the window
            mean = np.mean(current_values)
            std = np.std(current_values)
            if std < self.epsilon: # Avoid division by zero or near-zero std
                return 0.0 # Or value, or (value - mean) / epsilon
            return (value - mean) / (std + self.epsilon)
        
        elif self.strategy == 'min_max':
            # Use min/max from the current window
            min_v = np.min(current_values)
            max_v = np.max(current_values)
            if (max_v - min_v) < self.epsilon:
                return 0.0 # Or 0.5 if you want to center it
            return (value - min_v) / (max_v - min_v + self.epsilon)
        
        return value # Should not be reached if strategy is valid

    def reset(self):
        """Resets the normalizer's statistics."""
        self.values_history.clear()
        self.mean = 0.0
        self.std = 1.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.count = 0
        logger.debug("Normalizer has been reset.")

    def get_stats(self) -> Dict[str, Union[float, int]]:
        """Returns current normalization statistics."""
        if not self.values_history:
            return {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 0.0, "count": 0}
        
        current_values = np.array(list(self.values_history))
        return {
            "mean": np.mean(current_values),
            "std": np.std(current_values),
            "min": np.min(current_values),
            "max": np.max(current_values),
            "count": len(current_values)
        }