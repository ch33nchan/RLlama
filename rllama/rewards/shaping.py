import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal

DecaySchedule = Literal["none", "linear", "exponential", "cosine"]

@dataclass
class RewardConfig:
    name: str
    initial_weight: float
    decay_schedule: DecaySchedule = "none"
    min_weight: float = 0.0
    max_weight: float = 1.0 # Added for potential future use (e.g., adaptive scaling)
    # Renamed warmup_steps to decay_steps for clarity with schedules
    decay_steps: int = 1000
    # Added decay_rate for exponential schedule
    decay_rate: float = 0.99

class RewardShaper:
    def __init__(self, config: Dict[str, RewardConfig]):
        self.config = config
        # Initialize current weights and step counts
        self.current_weights = {name: cfg.initial_weight for name, cfg in config.items()}
        self.step_counts = {name: 0 for name in config}

    def update_weights(self, global_step: Optional[int] = None, metrics: Optional[Dict[str, float]] = None):
        """
        Updates the weights based on decay schedules and optionally adaptive metrics.
        Uses internal step counts if global_step is not provided.
        """
        for name, cfg in self.config.items():
            # Increment internal step count
            current_step = self.step_counts[name]
            self.step_counts[name] += 1

            # Use global_step if provided, otherwise use internal count
            effective_step = global_step if global_step is not None else current_step

            # Calculate target weight based on schedule
            target_weight = cfg.initial_weight
            if cfg.decay_schedule != "none" and cfg.decay_steps > 0:
                progress = min(1.0, effective_step / cfg.decay_steps) # Ensure progress doesn't exceed 1

                if cfg.decay_schedule == 'linear':
                    target_weight = cfg.initial_weight - (cfg.initial_weight - cfg.min_weight) * progress
                elif cfg.decay_schedule == 'exponential':
                    # decay_rate determines how fast it decays per step within decay_steps
                    # Effective decay rate adjusted for the number of steps
                    effective_decay_rate = cfg.decay_rate ** (1 / cfg.decay_steps) if cfg.decay_rate > 0 else 0
                    target_weight = cfg.initial_weight * (effective_decay_rate ** effective_step)
                    # Alternative: Simple exponential decay independent of decay_steps
                    # target_weight = self.current_weights[name] * cfg.decay_rate
                elif cfg.decay_schedule == 'cosine':
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    target_weight = cfg.min_weight + (cfg.initial_weight - cfg.min_weight) * cosine_decay

            # Ensure weight stays within bounds [min_weight, initial_weight] during decay
            # (max_weight is not used for decay clamping here, but could be for adaptive changes)
            self.current_weights[name] = max(cfg.min_weight, min(cfg.initial_weight, target_weight))

            # --- Placeholder for Adaptive Adjustment ---
            # This part can be expanded significantly based on metrics
            if metrics and name in metrics:
                 self._adjust_weight_from_metric(name, metrics[name])
            # --- End Placeholder ---


    def _adjust_weight_from_metric(self, name: str, metric: float):
        """Placeholder for dynamically adjusting weights based on performance metrics."""
        # Example: Very basic adjustment - increase weight slightly if metric is low
        cfg = self.config[name]
        current_weight = self.current_weights[name]
        if metric < 0.1: # Example threshold for poor performance
             self.current_weights[name] = min(cfg.max_weight, current_weight * 1.01)
        # More sophisticated logic needed here based on specific goals
        pass

    def get_weights(self) -> Dict[str, float]:
        """Returns the current weights."""
        return self.current_weights.copy()