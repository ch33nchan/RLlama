import logging
from typing import Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)

DecaySchedule = Literal['none', 'linear', 'exponential', 'linear_increase']

class RewardConfig:
    """Configuration for shaping a single reward component's weight over time."""
    def __init__(self,
                 name: str,
                 initial_weight: float = 1.0,
                 decay_schedule: DecaySchedule = 'none',
                 decay_rate: float = 0.0, # Rate for linear/exponential decay/increase
                 decay_steps: int = 0, # Total steps for decay/increase
                 min_weight: float = 0.0, # Floor for decay
                 max_weight: float = float('inf'), # Ceiling for increase
                 start_step: int = 0, # Step to start applying the schedule
                 **kwargs): # Allow extra params for future schedules
        self.name = name
        self.initial_weight = initial_weight
        self.decay_schedule = decay_schedule
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_weight = min_weight
        self.max_weight = max_weight # Add max_weight
        self.start_step = start_step # Add start_step
        self.current_weight = initial_weight
        self.extra_params = kwargs # Store any other params

        # Basic validation
        if decay_schedule != 'none' and decay_steps <= 0:
            logger.warning(f"RewardConfig '{name}': decay_schedule '{decay_schedule}' requires decay_steps > 0. Schedule disabled.")
            self.decay_schedule = 'none'
        if decay_schedule == 'linear_increase' and max_weight == float('inf'):
             logger.warning(f"RewardConfig '{name}': decay_schedule 'linear_increase' should ideally have a 'max_weight' defined.")


        logger.debug(f"Initialized RewardConfig '{name}': initial={initial_weight}, schedule={decay_schedule}, rate={decay_rate}, steps={decay_steps}, min={min_weight}, max={max_weight}, start={start_step}")

    def get_current_weight(self, current_step: int) -> float:
        """Calculates the weight at the current training step based on the schedule."""

        # Only apply schedule after start_step
        if current_step < self.start_step:
            return self.initial_weight # Or should it be 0 before start? Let's assume initial_weight.

        # Calculate effective steps *after* the start step
        effective_step = current_step - self.start_step

        if self.decay_schedule == 'none' or self.decay_steps <= 0:
            self.current_weight = self.initial_weight
        elif self.decay_schedule == 'linear':
            progress = min(effective_step / self.decay_steps, 1.0)
            weight_range = self.initial_weight - self.min_weight
            self.current_weight = self.initial_weight - (weight_range * progress)
            self.current_weight = max(self.current_weight, self.min_weight) # Ensure floor
        elif self.decay_schedule == 'exponential':
             # w_t = w_0 * (rate ^ (t / steps)) - needs careful rate definition
             # Alternative: w_t = w_min + (w_0 - w_min) * exp(-decay_rate * t / steps) ?
             # Let's use a simpler multiplicative decay: w_t = w_0 * (decay_rate ^ progress)
             # Requires decay_rate typically < 1, e.g., 0.999
             progress = min(effective_step / self.decay_steps, 1.0)
             # Ensure decay_rate is suitable for exponential decay (e.g., slightly less than 1)
             rate = self.decay_rate if 0 < self.decay_rate < 1 else 0.999 # Default if rate is invalid
             self.current_weight = self.initial_weight * (rate ** progress)
             self.current_weight = max(self.current_weight, self.min_weight) # Ensure floor
        elif self.decay_schedule == 'linear_increase':
            progress = min(effective_step / self.decay_steps, 1.0)
            # Calculate total increase possible (max_weight - initial_weight)
            # If max_weight is inf, use decay_rate to define slope? Let's assume max_weight is set.
            target_weight = self.max_weight if self.max_weight != float('inf') else self.initial_weight + self.decay_rate * self.decay_steps # Estimate target if max not set
            weight_range = target_weight - self.initial_weight
            self.current_weight = self.initial_weight + (weight_range * progress)
            self.current_weight = min(self.current_weight, self.max_weight) # Ensure ceiling

        return self.current_weight

class RewardShaper:
    """Manages multiple RewardConfig instances and provides current weights."""
    def __init__(self, component_reward_configs: Dict[str, RewardConfig], **kwargs): # MODIFIED signature
        self.configs: Dict[str, RewardConfig] = component_reward_configs # MODIFIED: Use passed configs directly
        self._current_step = 0
        
        # Handle known global shaper settings from kwargs
        self.composition_strategy = kwargs.get('composition_strategy', 'additive') # Default if not provided
        self.output_normalization_strategy = kwargs.get('output_normalization_strategy', 'none')
        self.output_norm_window = kwargs.get('output_norm_window', 100)
        self.output_epsilon = kwargs.get('output_epsilon', 1e-8)
        self.output_clip_range = kwargs.get('output_clip_range', None)
        
        logger.info(f"RewardShaper initialized with configs for: {list(self.configs.keys())}")
        logger.info(f"RewardShaper composition strategy: {self.composition_strategy}")
        logger.info(f"RewardShaper output normalization: {self.output_normalization_strategy}")
        
        # Log any other unexpected kwargs if necessary
        known_kwargs = {'composition_strategy', 'output_normalization_strategy', 'output_norm_window', 'output_epsilon', 'output_clip_range'}
        other_params = {k: v for k, v in kwargs.items() if k not in known_kwargs}
        if other_params:
            logger.info(f"RewardShaper received additional unhandled parameters: {other_params}")

    def update_weights(self, current_step: int):
        """Updates the internal step counter.""" # Docstring clarified
        self._current_step = current_step
        # Weights are calculated on demand in get_weights, so just update step

    def get_weights(self) -> Dict[str, float]:
        """Returns a dictionary of current weights for all managed components."""
        # Calls get_current_weight() on each managed RewardConfig
        return {name: config.get_current_weight(self._current_step)
                for name, config in self.configs.items()}

    # Optional: Add methods to update config parameters dynamically if needed
    # def update_config_param(self, component_name: str, param_name: str, value: Any): ...