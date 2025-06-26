import numpy as np
import torch
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import random

from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class AdaptiveClippingReward(BaseReward):
    """Automatically adjusts reward bounds using running statistics"""
    
    def __init__(self, initial_bounds: tuple = (-10.0, 10.0), adaptation_rate: float = 0.01,
                 percentile: float = 95, min_samples: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.percentile = percentile
        self.min_samples = min_samples
        self.lower_bound, self.upper_bound = initial_bounds
        self.reward_history = deque(maxlen=1000)
    
    def calculate(self, context: Dict[str, Any]) -> float:
        raw_reward = context.get('raw_reward', 0.0)
        
        # Add to history
        self.reward_history.append(raw_reward)
        
        # Update bounds if we have enough samples
        if len(self.reward_history) >= self.min_samples:
            rewards = np.array(self.reward_history)
            new_upper = np.percentile(rewards, self.percentile)
            new_lower = np.percentile(rewards, 100 - self.percentile)
            
            # Adapt bounds gradually
            self.upper_bound += self.adaptation_rate * (new_upper - self.upper_bound)
            self.lower_bound += self.adaptation_rate * (new_lower - self.lower_bound)
        
        # Clip the reward
        clipped_reward = np.clip(raw_reward, self.lower_bound, self.upper_bound)
        return clipped_reward
    
    def get_bounds(self) -> tuple:
        return (self.lower_bound, self.upper_bound)

@register_reward_component
class GradualCurriculumReward(BaseReward):
    """Implements curriculum learning with gradually increasing difficulty"""
    
    def __init__(self, initial_difficulty: float = 0.1, max_difficulty: float = 1.0,
                 progression_rate: float = 0.001, curriculum_schedule: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.progression_rate = progression_rate
        self.curriculum_schedule = curriculum_schedule
        self.current_difficulty = initial_difficulty
        self.step_count = 0
    
    def calculate(self, context: Dict[str, Any]) -> float:
        base_reward = context.get('base_reward', 0.0)
        task_difficulty = context.get('task_difficulty', 0.5)
        success_rate = context.get('success_rate', 0.0)
        
        self.step_count += 1
        
        # Update curriculum difficulty
        if self.curriculum_schedule == "linear":
            self.current_difficulty = min(
                self.max_difficulty,
                self.initial_difficulty + self.step_count * self.progression_rate
            )
        elif self.curriculum_schedule == "exponential":
            self.current_difficulty = min(
                self.max_difficulty,
                self.initial_difficulty * (1 + self.progression_rate) ** self.step_count
            )
        
        # Scale reward based on difficulty match
        difficulty_match = 1.0 - abs(task_difficulty - self.current_difficulty)
        curriculum_reward = base_reward * difficulty_match
        
        # Bonus for successful performance at current difficulty
        if success_rate > 0.8 and task_difficulty >= self.current_difficulty:
            curriculum_reward += 0.5
        
        return curriculum_reward
    
    def get_current_difficulty(self) -> float:
        return self.current_difficulty

@register_reward_component
class UncertaintyBasedReward(BaseReward):
    """Balances exploration and exploitation based on model uncertainty"""
    
    def __init__(self, exploration_bonus: float = 1.0, uncertainty_threshold: float = 0.5,
                 confidence_penalty: float = 0.1, exploration_decay: float = 0.999, **kwargs):
        super().__init__(**kwargs)
        self.exploration_bonus = exploration_bonus
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_penalty = confidence_penalty
        self.exploration_decay = exploration_decay
        self.step_count = 0
    
    def calculate(self, context: Dict[str, Any]) -> float:
        model_uncertainty = context.get('model_uncertainty', 0.0)
        confidence = context.get('confidence', 1.0)
        base_reward = context.get('base_reward', 0.0)
        
        self.step_count += 1
        
        # Decay exploration bonus over time
        current_exploration_bonus = self.exploration_bonus * (self.exploration_decay ** self.step_count)
        
        # Exploration reward for high uncertainty
        if model_uncertainty > self.uncertainty_threshold:
            exploration_reward = current_exploration_bonus * model_uncertainty
        else:
            exploration_reward = 0.0
        
        # Small penalty for overconfidence
        confidence_adjustment = -self.confidence_penalty * max(0, confidence - 0.9)
        
        return base_reward + exploration_reward + confidence_adjustment

@register_reward_component
class MetaLearningReward(BaseReward):
    """Rewards meta-cognitive strategies and fast adaptation"""
    
    def __init__(self, meta_bonus: float = 2.0, adaptation_weight: float = 1.5,
                 transfer_bonus: float = 1.0, learning_rate_sensitivity: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.meta_bonus = meta_bonus
        self.adaptation_weight = adaptation_weight
        self.transfer_bonus = transfer_bonus
        self.learning_rate_sensitivity = learning_rate_sensitivity
        self.previous_performance = 0.0
    
    def calculate(self, context: Dict[str, Any]) -> float:
        learning_rate = context.get('learning_rate', 0.01)
        adaptation_success = context.get('adaptation_success', False)
        transfer_performance = context.get('transfer_performance', 0.0)
        few_shot_accuracy = context.get('few_shot_accuracy', 0.0)
        
        total_reward = 0.0
        
        # Reward fast adaptation
        if adaptation_success:
            total_reward += self.meta_bonus
        
        # Reward transfer learning
        transfer_reward = transfer_performance * self.transfer_bonus
        total_reward += transfer_reward
        
        # Reward few-shot learning ability
        few_shot_reward = few_shot_accuracy * self.adaptation_weight
        total_reward += few_shot_reward
        
        # Adjust based on learning rate efficiency
        lr_efficiency = 1.0 / (1.0 + self.learning_rate_sensitivity * abs(learning_rate - 0.01))
        total_reward *= lr_efficiency
        
        return total_reward

@register_reward_component
class HindsightExperienceReward(BaseReward):
    """Learns from failures through retrospective goal relabeling"""
    
    def __init__(self, hindsight_bonus: float = 1.5, goal_tolerance: float = 0.1,
                 retrospective_weight: float = 1.0, goal_relabeling: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.hindsight_bonus = hindsight_bonus
        self.goal_tolerance = goal_tolerance
        self.retrospective_weight = retrospective_weight
        self.goal_relabeling = goal_relabeling
        self.failed_experiences = []
    
    def calculate(self, context: Dict[str, Any]) -> float:
        original_goal = context.get('original_goal')
        achieved_state = context.get('achieved_state')
        retrospective_goal = context.get('retrospective_goal')
        hindsight_success = context.get('hindsight_success', False)
        
        base_reward = 0.0
        
        # Standard reward for achieving original goal
        if original_goal is not None and achieved_state is not None:
            goal_distance = self._compute_goal_distance(original_goal, achieved_state)
            if goal_distance <= self.goal_tolerance:
                base_reward = 1.0
        
        # Hindsight experience replay bonus
        if self.goal_relabeling and hindsight_success:
            # Reward learning from relabeled experience
            hindsight_reward = self.hindsight_bonus * self.retrospective_weight
            base_reward += hindsight_reward
        
        return base_reward
    
    def _compute_goal_distance(self, goal, state) -> float:
        """Compute distance between goal and achieved state"""
        if isinstance(goal, str) and isinstance(state, str):
            return 0.0 if goal == state else 1.0
        return abs(hash(str(goal)) - hash(str(state))) / 1e10

@register_reward_component
class AdversarialReward(BaseReward):
    """Makes rewards robust through adversarial training"""
    
    def __init__(self, adversarial_strength: float = 0.1, robustness_bonus: float = 1.0,
                 perturbation_budget: float = 0.05, attack_type: str = "fgsm", **kwargs):
        super().__init__(**kwargs)
        self.adversarial_strength = adversarial_strength
        self.robustness_bonus = robustness_bonus
        self.perturbation_budget = perturbation_budget
        self.attack_type = attack_type
    
    def calculate(self, context: Dict[str, Any]) -> float:
        original_reward = context.get('original_reward', 0.0)
        perturbed_reward = context.get('perturbed_reward', original_reward)
        adversarial_perturbation = context.get('adversarial_perturbation', 0.0)
        robustness_score = context.get('robustness_score', 1.0)
        
        # Measure robustness as stability under perturbation
        reward_stability = 1.0 - abs(original_reward - perturbed_reward) / (abs(original_reward) + 1e-8)
        
        # Bonus for robustness to adversarial attacks
        if adversarial_perturbation <= self.perturbation_budget:
            robustness_reward = self.robustness_bonus * robustness_score
        else:
            robustness_reward = 0.0
        
        # Combine original reward with robustness bonus
        total_reward = original_reward + robustness_reward * reward_stability
        
        return total_reward

@register_reward_component
class RobustnessReward(BaseReward):
    """Ensures reward functions work reliably under various conditions"""
    
    def __init__(self, noise_tolerance: float = 0.2, stability_weight: float = 1.0,
                 consistency_bonus: float = 0.5, distribution_shift_penalty: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.noise_tolerance = noise_tolerance
        self.stability_weight = stability_weight
        self.consistency_bonus = consistency_bonus
        self.distribution_shift_penalty = distribution_shift_penalty
        self.reward_history = deque(maxlen=100)
    
    def calculate(self, context: Dict[str, Any]) -> float:
        base_reward = context.get('base_reward', 0.0)
        noise_level = context.get('noise_level', 0.0)
        perturbation_resistance = context.get('perturbation_resistance', 1.0)
        consistency_score = context.get('consistency_score', 1.0)
        stability_measure = context.get('stability_measure', 1.0)
        
        # Add to history for stability analysis
        self.reward_history.append(base_reward)
        
        # Calculate stability over recent history
        if len(self.reward_history) > 10:
            stability = 1.0 - np.std(list(self.reward_history)[-10:]) / (np.mean(list(self.reward_history)[-10:]) + 1e-8)
        else:
            stability = stability_measure
        
        # Robustness score based on multiple factors
        robustness_score = (
            perturbation_resistance * 0.4 +
            consistency_score * 0.3 +
            stability * 0.3
        )
        
        # Penalty for high noise levels
        noise_penalty = max(0, noise_level - self.noise_tolerance) * self.distribution_shift_penalty
        
        # Final robust reward
        robust_reward = base_reward * robustness_score - noise_penalty
        
        # Bonus for consistent performance
        if consistency_score > 0.9:
            robust_reward += self.consistency_bonus
        
        return robust_reward
