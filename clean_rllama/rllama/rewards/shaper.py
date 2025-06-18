# Replace the existing shaper.py with this enhanced version

from typing import Dict, List, Any, Optional, Union, Callable
import numpy as np
from enum import Enum

class ScheduleType(Enum):
    """Types of reward schedule functions"""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"
    CUSTOM = "custom"

class RewardConfig:
    """Configuration for a reward component"""
    
    def __init__(self, 
                 name: str,
                 weight: float = 1.0,
                 schedule_type: Union[ScheduleType, str] = ScheduleType.CONSTANT,
                 schedule_params: Dict[str, Any] = None,
                 min_weight: float = 0.0,
                 max_weight: float = float('inf'),
                 custom_schedule_fn: Optional[Callable[[int, Dict[str, Any]], float]] = None):
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
        """
        self.name = name
        self.base_weight = weight
        
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
    
    def get_weight(self, step: int) -> float:
        """
        Calculate the weight for a given step based on the schedule.
        
        Args:
            step: Current training step
            
        Returns:
            Weight value for the current step
        """
        if self.schedule_type == ScheduleType.CONSTANT:
            weight = self.base_weight
            
        elif self.schedule_type == ScheduleType.LINEAR:
            start_step = self.schedule_params.get("start_step", 0)
            end_step = self.schedule_params.get("end_step", 10000)
            start_value = self.schedule_params.get("start_value", self.base_weight)
            end_value = self.schedule_params.get("end_value", self.base_weight)
            
            if step <= start_step:
                weight = start_value
            elif step >= end_step:
                weight = end_value
            else:
                progress = (step - start_step) / (end_step - start_step)
                weight = start_value + progress * (end_value - start_value)
                
        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            decay_rate = self.schedule_params.get("decay_rate", 0.9999)
            start_step = self.schedule_params.get("start_step", 0)
            
            if step <= start_step:
                weight = self.base_weight
            else:
                decay_steps = step - start_step
                weight = self.base_weight * (decay_rate ** decay_steps)
                
        elif self.schedule_type == ScheduleType.COSINE:
            start_step = self.schedule_params.get("start_step", 0)
            end_step = self.schedule_params.get("end_step", 10000)
            start_value = self.schedule_params.get("start_value", self.base_weight)
            end_value = self.schedule_params.get("end_value", 0.0)
            
            if step <= start_step:
                weight = start_value
            elif step >= end_step:
                weight = end_value
            else:
                progress = (step - start_step) / (end_step - start_step)
                cosine_term = 0.5 * (1 + np.cos(np.pi * progress))
                weight = end_value + (start_value - end_value) * cosine_term
                
        elif self.schedule_type == ScheduleType.STEP:
            boundaries = self.schedule_params.get("boundaries", [1000, 2000, 3000])
            values = self.schedule_params.get("values", [self.base_weight, self.base_weight * 0.5, self.base_weight * 0.1, self.base_weight * 0.01])
            
            if len(values) != len(boundaries) + 1:
                raise ValueError("Step schedule needs len(values) == len(boundaries) + 1")
                
            weight = values[0]
            for i, boundary in enumerate(boundaries):
                if step >= boundary:
                    weight = values[i + 1]
        
        elif self.schedule_type == ScheduleType.CUSTOM and self.custom_schedule_fn:
            weight = self.custom_schedule_fn(step, self.schedule_params)
            
        else:
            weight = self.base_weight
        
        # Clamp weight to allowed range
        return max(self.min_weight, min(self.max_weight, weight))


class RewardShaper:
    """
    Applies weights from a configuration to a dictionary of raw
    component rewards to produce a final, single scalar reward.
    
    This class handles advanced reward shaping, such as scheduling,
    where weights change over time.
    """
    def __init__(self, shaping_config: Dict):
        """
        Args:
            shaping_config (Dict): The 'shaping_config' block from your
                                   YAML file.
        """
        self.config = shaping_config
        
        # Parse configurations
        self.reward_configs = {}
        for name, config in shaping_config.items():
            if isinstance(config, dict):
                weight = config.get("weight", 1.0)
                schedule_type = config.get("schedule", "constant")
                schedule_params = config.get("schedule_params", {})
                min_weight = config.get("min_weight", 0.0)
                max_weight = config.get("max_weight", float('inf'))
                
                self.reward_configs[name] = RewardConfig(
                    name=name,
                    weight=weight,
                    schedule_type=schedule_type,
                    schedule_params=schedule_params,
                    min_weight=min_weight,
                    max_weight=max_weight
                )
            else:
                # Simple case: just a weight value
                self.reward_configs[name] = RewardConfig(
                    name=name,
                    weight=float(config)
                )

    def shape(self, component_rewards: Dict[str, float], step: int = 0) -> float:
        """
        Shapes the final reward by applying weights and summing the components.

        Args:
            component_rewards (Dict[str, float]): A dict of raw reward values
                                                 from the RewardComposer.
            step (int): The current training step, for scheduling.

        Returns:
            The final shaped reward as a single float.
        """
        final_reward = 0.0
        
        for name, reward_val in component_rewards.items():
            # Get weight from config, default to 1.0 if the component is
            # not listed in the shaping_config block.
            if name in self.reward_configs:
                weight = self.reward_configs[name].get_weight(step)
            else:
                # Default to weight 1.0 for components not in config
                weight = 1.0
                
            final_reward += weight * reward_val

        return final_reward
    
    def get_current_weights(self, step: int = 0) -> Dict[str, float]:
        """
        Get the current weights for all components at a specific step.
        
        Args:
            step: The current training step
            
        Returns:
            Dictionary mapping component names to their current weights
        """
        return {name: config.get_weight(step) for name, config in self.reward_configs.items()}
    
    def add_component(self, name: str, config: Union[float, Dict[str, Any], RewardConfig]):
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
                max_weight=config.get("max_weight", float('inf'))
            )
        else:
            self.reward_configs[name] = RewardConfig(
                name=name,
                weight=float(config)
            )