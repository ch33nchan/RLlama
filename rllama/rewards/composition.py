import numpy as np
from typing import Dict, List, Callable, Any, Optional, Union
import math

class RewardComposer:
    def __init__(self, components: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.components = components
        self.weights = weights or {name: 1.0 for name in components.keys()}
        self.composition_method = "linear"
        self.composition_params = {}
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        for name, weight in weights.items():
            if name in self.weights:
                self.weights[name] = weight
    
    def set_composition_method(self, method: str, **params) -> None:
        self.composition_method = method
        self.composition_params = params
    
    def calculate(self, state: Any, action: Any) -> float:
        component_values = {}
        
        for name, component in self.components.items():
            if name in self.weights:
                component_values[name] = self.weights[name] * component.calculate(state, action)
        
        return self._compose_rewards(component_values)
    
    def _compose_rewards(self, component_values: Dict[str, float]) -> float:
        if self.composition_method == "linear":
            return sum(component_values.values())
        
        elif self.composition_method == "multiplicative":
            result = 1.0
            for value in component_values.values():
                result *= (1.0 + value)
            return result - 1.0
        
        elif self.composition_method == "min":
            if not component_values:
                return 0.0
            return min(component_values.values())
        
        elif self.composition_method == "max":
            if not component_values:
                return 0.0
            return max(component_values.values())
        
        elif self.composition_method == "geometric_mean":
            if not component_values:
                return 0.0
            values = list(component_values.values())
            
            offset = self.composition_params.get("offset", 0.0)
            adjusted_values = [v + offset for v in values]
            
            if any(v <= 0 for v in adjusted_values):
                return min(values)
            
            return math.pow(np.prod(adjusted_values), 1.0 / len(adjusted_values)) - offset
        
        elif self.composition_method == "priority":
            priority_order = self.composition_params.get("priority_order", list(component_values.keys()))
            for name in priority_order:
                if name in component_values:
                    threshold = self.composition_params.get("thresholds", {}).get(name, 0.0)
                    if component_values[name] < threshold:
                        return component_values[name]
            return sum(component_values.values())
        
        elif self.composition_method == "conditional":
            conditions = self.composition_params.get("conditions", [])
            for condition in conditions:
                component_name = condition.get("component")
                threshold = condition.get("threshold", 0.0)
                comparison = condition.get("comparison", "less_than")
                
                if component_name in component_values:
                    value = component_values[component_name]
                    
                    if comparison == "less_than" and value < threshold:
                        return condition.get("return_value", value)
                    elif comparison == "greater_than" and value > threshold:
                        return condition.get("return_value", value)
                    elif comparison == "equal" and value == threshold:
                        return condition.get("return_value", value)
            
            return sum(component_values.values())
        
        elif self.composition_method == "softmax":
            temperature = self.composition_params.get("temperature", 1.0)
            values = list(component_values.values())
            if not values:
                return 0.0
                
            max_val = max(values)
            exp_values = [math.exp((v - max_val) / temperature) for v in values]
            softmax_weights = [ev / sum(exp_values) for ev in exp_values]
            
            return sum(w * v for w, v in zip(softmax_weights, values))
        
        else:
            return sum(component_values.values())

class DynamicRewardComposer(RewardComposer):
    def __init__(self, components: Dict[str, Any], 
                 weight_schedulers: Optional[Dict[str, Callable[[int], float]]] = None):
        super().__init__(components)
        self.weight_schedulers = weight_schedulers or {}
        self.step_counter = 0
    
    def update_step(self) -> None:
        self.step_counter += 1
        self._update_weights()
    
    def _update_weights(self) -> None:
        for name, scheduler in self.weight_schedulers.items():
            if name in self.weights:
                self.weights[name] = scheduler(self.step_counter)
    
    def calculate(self, state: Any, action: Any) -> float:
        self.update_step()
        return super().calculate(state, action)

class HierarchicalRewardComposer:
    def __init__(self, level_composers: List[RewardComposer], 
                 level_weights: Optional[List[float]] = None):
        self.level_composers = level_composers
        self.level_weights = level_weights or [1.0] * len(level_composers)
    
    def calculate(self, state: Any, action: Any) -> float:
        level_rewards = []
        
        for i, composer in enumerate(self.level_composers):
            level_reward = composer.calculate(state, action)
            weighted_reward = self.level_weights[i] * level_reward
            level_rewards.append(weighted_reward)
        
        return sum(level_rewards)
    
    def set_level_weights(self, weights: List[float]) -> None:
        if len(weights) == len(self.level_composers):
            self.level_weights = weights