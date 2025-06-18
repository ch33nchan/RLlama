# Add this class to the existing normalization.py file

class AdaptiveNormalizer(RewardNormalizer):
    """
    Adaptive normalizer that adjusts normalization based on the distribution of rewards.
    Uses a combination of methods to handle different reward distributions.
    """
    
    def __init__(self, 
                 initial_method: str = "standard", 
                 window_size: int = 100,
                 adaptation_threshold: int = 50,
                 scale: float = 1.0):
        """
        Initialize the adaptive normalizer.
        
        Args:
            initial_method: Initial normalization method to use.
            window_size: Size of the sliding window for recent values.
            adaptation_threshold: Number of samples after which to start adapting.
            scale: Scale factor for normalization.
        """
        super().__init__(method=initial_method, window_size=window_size)
        self.adaptation_threshold = adaptation_threshold
        self.scale = scale
        self.methods = ["standard", "minmax", "robust", "tanh", "percentile"]
        self.method_weights = {method: 1.0 for method in self.methods}
        self.method_errors = {method: [] for method in self.methods}
    
    def normalize(self, reward: float, component_name: str = "default") -> float:
        """
        Normalize a reward value using adaptive normalization.
        
        Args:
            reward: The raw reward value to normalize.
            component_name: The name of the reward component.
            
        Returns:
            The normalized reward value.
        """
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
        
        # Update statistics
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
        
        # If we have enough samples, adapt normalization method
        if stats["count"] >= self.adaptation_threshold:
            return self._adaptive_normalize(reward, stats, component_name)
        else:
            # Use the initial method for the first few samples
            return self._normalize_with_method(reward, stats, self.method)
    
    def _normalize_with_method(self, reward: float, stats: dict, method: str) -> float:
        """Normalize a reward using a specific method"""
        if method == "standard":
            if stats["std"] > 0:
                return (reward - stats["mean"]) / stats["std"]
            else:
                return 0.0
        
        elif method == "minmax":
            if stats["max"] > stats["min"]:
                return (reward - stats["min"]) / (stats["max"] - stats["min"])
            else:
                return 0.0
        
        elif method == "percentile":
            if len(stats["recent_values"]) > 1:
                sorted_values = sorted(stats["recent_values"])
                rank = sorted_values.index(reward) if reward in sorted_values else 0
                return rank / (len(sorted_values) - 1)
            else:
                return 0.0
        
        elif method == "robust":
            if len(stats["recent_values"]) >= 5:  # Need enough samples for percentiles
                q1 = np.percentile(list(stats["recent_values"]), 25)
                q3 = np.percentile(list(stats["recent_values"]), 75)
                iqr = q3 - q1
                if iqr > 0:
                    return (reward - stats["mean"]) / iqr
                else:
                    return (reward - stats["mean"]) / (stats["std"] if stats["std"] > 0 else 1.0)
            else:
                return (reward - stats["mean"]) / (stats["std"] if stats["std"] > 0 else 1.0)
        
        elif method == "tanh":
            return np.tanh(reward / self.scale)
        
        else:  # Default or unknown method
            return reward
    
    def _adaptive_normalize(self, reward: float, stats: dict, component_name: str) -> float:
        """Use adaptive normalization to choose the best method"""
        # Get distribution characteristics
        skewness = self._calculate_skewness(list(stats["recent_values"]))
        kurtosis = self._calculate_kurtosis(list(stats["recent_values"]))
        
        # Determine which method might be best based on distribution
        if abs(skewness) > 2.0:  # Highly skewed
            if reward > stats["mean"] + 3 * stats["std"] or reward < stats["mean"] - 3 * stats["std"]:
                # Outlier, use robust method
                self.method = "robust"
            else:
                # Skewed but not outlier, prefer minmax or tanh
                if kurtosis > 3.0:  # Heavy tailed
                    self.method = "tanh"
                else:
                    self.method = "minmax"
        else:  # Not heavily skewed
            if kurtosis > 4.0:  # Heavy tailed
                self.method = "percentile"
            else:  # Close to normal
                self.method = "standard"
        
        # Apply the selected normalization method
        normalized_value = self._normalize_with_method(reward, stats, self.method)
        
        # Constrain the result to a reasonable range
        return max(-5.0, min(5.0, normalized_value))
    
    def _calculate_skewness(self, values: list) -> float:
        """Calculate the skewness of a distribution"""
        if not values or len(values) < 3:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        if variance == 0:
            return 0.0
        
        std_dev = np.sqrt(variance)
        skewness = sum((x - mean) ** 3 for x in values) / (len(values) * std_dev ** 3)
        return skewness
    
    def _calculate_kurtosis(self, values: list) -> float:
        """Calculate the kurtosis of a distribution"""
        if not values or len(values) < 4:
            return 3.0  # Normal distribution kurtosis
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        if variance == 0:
            return 3.0
        
        std_dev = np.sqrt(variance)
        kurtosis = sum((x - mean) ** 4 for x in values) / (len(values) * std_dev ** 4)
        return kurtosis