#!/usr/bin/env python3
"""
Adversarial reward components that implement adversarial training and robustness mechanisms.
These components help create more robust reward functions through adversarial examples and training.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Union, List, Deque, Callable
from collections import deque, defaultdict
import time

from ...base import BaseReward
from ...registry import register_reward_component

@register_reward_component
class AdversarialReward(BaseReward):
    """
    Reward component that implements adversarial training for reward robustness.
    Generates adversarial examples and adjusts rewards based on worst-case scenarios.
    """
    
    def __init__(self,
                 base_reward_key: str = "base_reward",
                 adversarial_strength: float = 0.1,
                 perturbation_type: str = "gaussian",
                 num_adversarial_samples: int = 5,
                 robustness_weight: float = 0.3,
                 adaptation_rate: float = 0.01,
                 min_perturbation: float = 0.01,
                 max_perturbation: float = 0.5,
                 **kwargs):
        """
        Initialize adversarial reward component.
        
        Args:
            base_reward_key: Key in context containing base reward to make adversarial
            adversarial_strength: Strength of adversarial perturbations
            perturbation_type: Type of perturbation ("gaussian", "uniform", "targeted")
            num_adversarial_samples: Number of adversarial samples to generate
            robustness_weight: Weight for robustness term in final reward
            adaptation_rate: Rate of adaptation for adversarial strength
            min_perturbation: Minimum perturbation strength
            max_perturbation: Maximum perturbation strength
        """
        super().__init__(**kwargs)
        self.base_reward_key = base_reward_key
        self.adversarial_strength = adversarial_strength
        self.initial_strength = adversarial_strength
        self.perturbation_type = perturbation_type
        self.num_adversarial_samples = num_adversarial_samples
        self.robustness_weight = robustness_weight
        self.adaptation_rate = adaptation_rate
        self.min_perturbation = min_perturbation
        self.max_perturbation = max_perturbation
        
        # Track adversarial statistics
        self.adversarial_history: Deque = deque(maxlen=1000)
        self.success_rate = 0.0
        self.step_count = 0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate adversarial reward with robustness considerations."""
        # Extract base reward
        base_reward = context.get(self.base_reward_key, 0.0)
        if not isinstance(base_reward, (int, float)):
            return 0.0
            
        # Extract features for adversarial perturbation
        features = self._extract_features(context)
        if features is None:
            return base_reward
            
        # Generate adversarial examples
        adversarial_rewards = self._generate_adversarial_examples(features, context)
        
        # Calculate robustness metrics
        if adversarial_rewards:
            min_adversarial = min(adversarial_rewards)
            max_adversarial = max(adversarial_rewards)
            mean_adversarial = np.mean(adversarial_rewards)
            std_adversarial = np.std(adversarial_rewards)
            
            # Robustness penalty based on worst-case performance
            robustness_penalty = max(0, base_reward - min_adversarial)
            
            # Stability bonus for consistent performance
            stability_bonus = 1.0 / (1.0 + std_adversarial)
            
            # Final adversarial reward
            adversarial_reward = (
                (1 - self.robustness_weight) * base_reward +
                self.robustness_weight * (mean_adversarial + stability_bonus - robustness_penalty)
            )
            
            # Track statistics
            self.adversarial_history.append({
                'base_reward': base_reward,
                'adversarial_rewards': adversarial_rewards,
                'robustness_penalty': robustness_penalty,
                'stability_bonus': stability_bonus
            })
            
        else:
            adversarial_reward = base_reward
            
        # Adapt adversarial strength based on performance
        self._adapt_adversarial_strength()
        
        self.step_count += 1
        return adversarial_reward
        
    def _extract_features(self, context: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features for adversarial perturbation."""
        # Try to extract numerical features from context
        features = []
        
        # Look for common feature keys
        feature_keys = ['observation', 'state', 'features', 'embedding']
        
        for key in feature_keys:
            if key in context:
                value = context[key]
                if isinstance(value, (list, tuple)):
                    features.extend([float(x) for x in value if isinstance(x, (int, float))])
                elif isinstance(value, (int, float)):
                    features.append(float(value))
                elif hasattr(value, 'flatten'):  # numpy array or tensor
                    try:
                        flat = value.flatten()
                        features.extend([float(x) for x in flat])
                    except:
                        continue
                        
        if features:
            return np.array(features)
        else:
            # Fallback: create features from text if available
            text_keys = ['response', 'text', 'content']
            for key in text_keys:
                if key in context and isinstance(context[key], str):
                    text = context[key]
                    # Simple text features: length, word count, character frequencies
                    features = [
                        len(text),
                        len(text.split()),
                        text.count(' ') / max(1, len(text)),
                        text.count('.') / max(1, len(text)),
                        sum(1 for c in text if c.isupper()) / max(1, len(text))
                    ]
                    return np.array(features)
                    
        return None
        
    def _generate_adversarial_examples(self, 
                                     features: np.ndarray, 
                                     context: Dict[str, Any]) -> List[float]:
        """Generate adversarial examples and compute their rewards."""
        adversarial_rewards = []
        
        for _ in range(self.num_adversarial_samples):
            # Generate perturbation
            if self.perturbation_type == "gaussian":
                perturbation = np.random.normal(0, self.adversarial_strength, features.shape)
            elif self.perturbation_type == "uniform":
                perturbation = np.random.uniform(
                    -self.adversarial_strength, 
                    self.adversarial_strength, 
                    features.shape
                )
            elif self.perturbation_type == "targeted":
                # Targeted perturbation towards worst-case direction
                if hasattr(self, 'worst_direction') and self.worst_direction is not None:
                    perturbation = self.adversarial_strength * self.worst_direction
                else:
                    perturbation = np.random.normal(0, self.adversarial_strength, features.shape)
            else:
                perturbation = np.random.normal(0, self.adversarial_strength, features.shape)
                
            # Apply perturbation
            perturbed_features = features + perturbation
            
            # Create adversarial context
            adversarial_context = context.copy()
            
            # Update context with perturbed features
            # This is a simplified approach - in practice, you'd map back to original space
            if 'observation' in context:
                adversarial_context['observation'] = perturbed_features
            elif 'state' in context:
                adversarial_context['state'] = perturbed_features
            elif 'features' in context:
                adversarial_context['features'] = perturbed_features
                
            # Compute adversarial reward (simplified - would use actual reward function)
            adversarial_reward = self._compute_perturbed_reward(
                adversarial_context, perturbation
            )
            adversarial_rewards.append(adversarial_reward)
            
        return adversarial_rewards
        
    def _compute_perturbed_reward(self, 
                                adversarial_context: Dict[str, Any], 
                                perturbation: np.ndarray) -> float:
        """Compute reward for perturbed context."""
        # Get base reward
        base_reward = adversarial_context.get(self.base_reward_key, 0.0)
        
        # Apply perturbation effect (simplified model)
        perturbation_magnitude = np.linalg.norm(perturbation)
        
        # Assume perturbation generally decreases reward
        perturbation_effect = -perturbation_magnitude * 0.1
        
        # Add some randomness to simulate complex reward landscape
        noise = np.random.normal(0, 0.05)
        
        return base_reward + perturbation_effect + noise
        
    def _adapt_adversarial_strength(self) -> None:
        """Adapt adversarial strength based on success rate."""
        if len(self.adversarial_history) < 10:
            return
            
        # Calculate recent success rate (how often adversarial examples are worse)
        recent_history = list(self.adversarial_history)[-10:]
        successes = 0
        
        for entry in recent_history:
            base_reward = entry['base_reward']
            adversarial_rewards = entry['adversarial_rewards']
            
            # Success if any adversarial example is significantly worse
            if any(adv_r < base_reward - 0.1 for adv_r in adversarial_rewards):
                successes += 1
                
        self.success_rate = successes / len(recent_history)
        
        # Adapt strength based on success rate
        target_success_rate = 0.3  # Want some adversarial examples to be successful
        
        if self.success_rate < target_success_rate:
            # Increase strength if not finding enough adversarial examples
            self.adversarial_strength = min(
                self.max_perturbation,
                self.adversarial_strength * (1 + self.adaptation_rate)
            )
        elif self.success_rate > target_success_rate * 1.5:
            # Decrease strength if too many adversarial examples
            self.adversarial_strength = max(
                self.min_perturbation,
                self.adversarial_strength * (1 - self.adaptation_rate)
            )
            
    def get_statistics(self) -> Dict[str, float]:
        """Get adversarial training statistics."""
        if not self.adversarial_history:
            return {}
            
        recent_entries = list(self.adversarial_history)[-100:]
        
        avg_robustness_penalty = np.mean([
            entry['robustness_penalty'] for entry in recent_entries
        ])
        
        avg_stability_bonus = np.mean([
            entry['stability_bonus'] for entry in recent_entries
        ])
        
        return {
            'adversarial_strength': self.adversarial_strength,
            'success_rate': self.success_rate,
            'avg_robustness_penalty': avg_robustness_penalty,
            'avg_stability_bonus': avg_stability_bonus,
            'total_samples': len(self.adversarial_history)
        }

@register_reward_component
class RobustnessReward(BaseReward):
    """
    Reward component that measures and encourages robustness to input variations.
    Focuses on consistency across similar inputs and graceful degradation.
    """
    
    def __init__(self,
                 input_key: str = "input",
                 sensitivity_threshold: float = 0.1,
                 robustness_weight: float = 0.5,
                 perturbation_scale: float = 0.05,
                 consistency_window: int = 10,
                 **kwargs):
        """
        Initialize robustness reward component.
        
        Args:
            input_key: Key in context containing input to test robustness
            sensitivity_threshold: Threshold for considering changes significant
            robustness_weight: Weight for robustness in final reward
            perturbation_scale: Scale of perturbations for robustness testing
            consistency_window: Window size for consistency measurement
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.sensitivity_threshold = sensitivity_threshold
        self.robustness_weight = robustness_weight
        self.perturbation_scale = perturbation_scale
        self.consistency_window = consistency_window
        
        # Track input-output pairs for robustness analysis
        self.input_history: Deque = deque(maxlen=1000)
        self.output_history: Deque = deque(maxlen=1000)
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate robustness-based reward."""
        # Extract input and base reward
        input_data = context.get(self.input_key)
        base_reward = context.get('base_reward', 0.0)
        
        if input_data is None:
            return base_reward
            
        # Convert input to numerical representation
        input_vector = self._vectorize_input(input_data)
        if input_vector is None:
            return base_reward
            
        # Store current input-output pair
        self.input_history.append(input_vector)
        self.output_history.append(base_reward)
        
        # Calculate robustness metrics
        robustness_score = self._calculate_robustness(input_vector, base_reward)
        
        # Combine base reward with robustness
        final_reward = (
            (1 - self.robustness_weight) * base_reward +
            self.robustness_weight * robustness_score
        )
        
        return final_reward
        
    def _vectorize_input(self, input_data: Any) -> Optional[np.ndarray]:
        """Convert input data to numerical vector."""
        if isinstance(input_data, (list, tuple)):
            try:
                return np.array([float(x) for x in input_data])
            except (ValueError, TypeError):
                return None
        elif isinstance(input_data, str):
            # Simple text vectorization
            return np.array([
                len(input_data),
                len(input_data.split()),
                input_data.count(' ') / max(1, len(input_data)),
                sum(1 for c in input_data if c.isupper()) / max(1, len(input_data)),
                sum(1 for c in input_data if c.islower()) / max(1, len(input_data))
            ])
        elif hasattr(input_data, 'flatten'):
            try:
                return input_data.flatten()
            except:
                return None
        elif isinstance(input_data, (int, float)):
            return np.array([float(input_data)])
        else:
            return None
            
    def _calculate_robustness(self, 
                            current_input: np.ndarray, 
                            current_output: float) -> float:
        """Calculate robustness score based on input-output consistency."""
        if len(self.input_history) < 2:
            return 0.5  # Neutral score for insufficient data
            
        # Find similar inputs in history
        similarities = []
        output_differences = []
        
        for i, (hist_input, hist_output) in enumerate(
            zip(list(self.input_history)[:-1], list(self.output_history)[:-1])
        ):
            # Calculate input similarity (cosine similarity)
            if hist_input.shape == current_input.shape:
                similarity = self._cosine_similarity(current_input, hist_input)
                similarities.append(similarity)
                
                # Calculate output difference
                output_diff = abs(current_output - hist_output)
                output_differences.append(output_diff)
                
        if not similarities:
            return 0.5
            
        # Calculate robustness metrics
        similarities = np.array(similarities)
        output_differences = np.array(output_differences)
        
        # Find highly similar inputs (similarity > 0.8)
        high_similarity_mask = similarities > 0.8
        
        if np.any(high_similarity_mask):
            # For highly similar inputs, outputs should be similar
            similar_output_diffs = output_differences[high_similarity_mask]
            consistency_score = 1.0 / (1.0 + np.mean(similar_output_diffs))
        else:
            consistency_score = 0.5
            
        # Calculate sensitivity score (how much output changes with input changes)
        if len(similarities) > 5:
            # Use recent samples for sensitivity analysis
            recent_sims = similarities[-5:]
            recent_diffs = output_differences[-5:]
            
            # Good robustness: small input changes -> small output changes
            sensitivity_scores = []
            for sim, diff in zip(recent_sims, recent_diffs):
                input_change = 1.0 - sim  # How much input changed
                if input_change > 0:
                    sensitivity = diff / input_change
                    sensitivity_scores.append(sensitivity)
                    
            if sensitivity_scores:
                avg_sensitivity = np.mean(sensitivity_scores)
                sensitivity_score = 1.0 / (1.0 + avg_sensitivity)
            else:
                sensitivity_score = 0.5
        else:
            sensitivity_score = 0.5
            
        # Combine consistency and sensitivity
        robustness_score = 0.6 * consistency_score + 0.4 * sensitivity_score
        
        return robustness_score
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
        
    def get_robustness_analysis(self) -> Dict[str, Any]:
        """Get detailed robustness analysis."""
        if len(self.input_history) < 5:
            return {"error": "Insufficient data for analysis"}
            
        # Analyze input-output relationships
        inputs = np.array(list(self.input_history))
        outputs = np.array(list(self.output_history))
        
        # Calculate pairwise similarities and output differences
        n_samples = len(inputs)
        similarities = []
        output_diffs = []
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if inputs[i].shape == inputs[j].shape:
                    sim = self._cosine_similarity(inputs[i], inputs[j])
                    diff = abs(outputs[i] - outputs[j])
                    similarities.append(sim)
                    output_diffs.append(diff)
                    
        if not similarities:
            return {"error": "No comparable inputs found"}
            
        similarities = np.array(similarities)
        output_diffs = np.array(output_diffs)
        
        # Robustness metrics
        high_sim_mask = similarities > 0.8
        medium_sim_mask = (similarities > 0.5) & (similarities <= 0.8)
        low_sim_mask = similarities <= 0.5
        
        analysis = {
            "total_comparisons": len(similarities),
            "high_similarity_pairs": np.sum(high_sim_mask),
            "medium_similarity_pairs": np.sum(medium_sim_mask),
            "low_similarity_pairs": np.sum(low_sim_mask)
        }
        
        if np.any(high_sim_mask):
            analysis["high_sim_output_consistency"] = {
                "mean_output_diff": np.mean(output_diffs[high_sim_mask]),
                "std_output_diff": np.std(output_diffs[high_sim_mask]),
                "max_output_diff": np.max(output_diffs[high_sim_mask])
            }
            
        # Overall robustness score
        if np.any(high_sim_mask):
            consistency = 1.0 / (1.0 + np.mean(output_diffs[high_sim_mask]))
        else:
            consistency = 0.5
            
        analysis["overall_robustness_score"] = consistency
        
        return analysis
