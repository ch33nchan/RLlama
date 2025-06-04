# rllama/integration/trl_wrapper.py

import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import re
from collections import Counter

class TRLRllamaRewardProcessor:
    """
    Complete TRL integration for RLlama reward processing
    """
    
    def __init__(self, rllama_config_path: str, auto_register_llm_components: bool = True):
        self.config_path = rllama_config_path
        self.step_count = 0
        self.component_history = []
        self.load_config()
        print("TRLRllamaRewardProcessor initialized successfully.")
        print(f"  Loaded {len(self.components)} reward components")
        
    def load_config(self):
        """Load YAML config and parse components"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.components = self.config.get('composer', {}).get('components', [])
            self.shaper_config = self.config.get('shaper', {})
            self.normalization_method = self.shaper_config.get('normalization_method', 'none')
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {}
            self.components = []
            self.shaper_config = {}
            
    def compute_rewards(self, prompts_text: List[str], responses_text: List[str], 
                       model_specific_infos: Optional[Any] = None) -> List[float]:
        """
        Main method to compute rewards for prompt-response pairs
        """
        if not prompts_text or not responses_text:
            return []
            
        batch_rewards = []
        component_details = {}
        
        # Initialize component tracking
        for component in self.components:
            component_details[component.get('type', 'unknown')] = []
        
        # Process each prompt-response pair
        for prompt, response in zip(prompts_text, responses_text):
            total_reward = 0.0
            
            # Calculate each component reward
            for component in self.components:
                component_type = component.get('type', '')
                base_weight = component.get('weight', 1.0)
                
                # Apply scheduling if configured
                current_weight = self._apply_scheduling(component, base_weight)
                
                # Calculate component-specific reward
                component_reward = self._calculate_component_reward(
                    component_type, prompt, response
                )
                
                weighted_reward = current_weight * component_reward
                total_reward += weighted_reward
                
                # Track for analysis
                component_details[component_type].append(weighted_reward)
            
            batch_rewards.append(total_reward)
        
        # Apply normalization
        normalized_rewards = self._apply_normalization(batch_rewards)
        
        # Store for analysis
        self.component_history.append({
            'step': self.step_count,
            'components': component_details,
            'raw_rewards': batch_rewards,
            'normalized_rewards': normalized_rewards
        })
        
        self.step_count += 1
        return normalized_rewards
    
    def _apply_scheduling(self, component: Dict, base_weight: float) -> float:
        """Apply weight scheduling based on component configuration"""
        schedule = component.get('schedule', {})
        if not schedule:
            return base_weight
            
        schedule_type = schedule.get('type', 'constant')
        
        if schedule_type == 'exponential_decay':
            decay_rate = schedule.get('decay_rate', 0.95)
            return base_weight * (decay_rate ** self.step_count)
        elif schedule_type == 'linear_decay':
            decay_steps = schedule.get('decay_steps', 100)
            if self.step_count >= decay_steps:
                return 0.0
            return base_weight * (1.0 - self.step_count / decay_steps)
        
        return base_weight
    
    def _calculate_component_reward(self, component_type: str, prompt: str, response: str) -> float:
        """Calculate reward for specific component types"""
        
        if component_type == "CoherenceReward":
            return self._coherence_reward(prompt, response)
        elif component_type == "HelpfulnessReward":
            return self._helpfulness_reward(prompt, response)
        elif component_type == "DiversityReward":
            return self._diversity_reward(prompt, response)
        elif component_type == "ConcisenessReward":
            return self._conciseness_reward(prompt, response)
        elif component_type == "FactualityReward":
            return self._factuality_reward(prompt, response)
        elif component_type == "LengthReward":
            return self._length_reward(prompt, response)
        elif component_type == "EntropyBonus":
            return self._entropy_bonus(prompt, response)
        else:
            return 0.0
    
    def _coherence_reward(self, prompt: str, response: str) -> float:
        """Reward coherent, well-structured responses"""
        score = 0.0
        
        # Check for proper sentence structure
        sentences = response.split('.')
        if len(sentences) > 1:
            score += 0.3
        
        # Penalize very short responses
        if len(response.split()) < 5:
            score -= 0.3
        
        # Reward medium-length responses
        word_count = len(response.split())
        if 10 <= word_count <= 50:
            score += 0.5
        
        return np.clip(score, 0.0, 1.0)
    
    def _helpfulness_reward(self, prompt: str, response: str) -> float:
        """Reward responses that address the prompt"""
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        overlap = len(prompt_words.intersection(response_words))
        relevance_score = overlap / max(len(prompt_words), 1)
        
        # Bonus for question answering
        if '?' in prompt and any(word in response.lower() for word in ['because', 'since', 'due to', 'is', 'are']):
            relevance_score += 0.3
        
        return np.clip(relevance_score, 0.0, 1.0)
    
    def _diversity_reward(self, prompt: str, response: str) -> float:
        """Reward lexical diversity"""
        words = response.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        diversity_ratio = unique_words / total_words
        
        return np.clip(diversity_ratio, 0.0, 1.0)
    
    def _conciseness_reward(self, prompt: str, response: str) -> float:
        """Reward concise but complete responses"""
        word_count = len(response.split())
        
        if word_count < 5:
            return 0.1  # Too short
        elif 5 <= word_count <= 25:
            return 1.0  # Optimal length
        elif 25 < word_count <= 50:
            return 0.7  # Acceptable
        else:
            return max(0.2, 1.0 - (word_count - 50) / 100)  # Penalize verbosity
    
    def _factuality_reward(self, prompt: str, response: str) -> float:
        """Basic factuality assessment"""
        # Look for factual indicators
        factual_patterns = [
            r'\d{4}',  # Years
            r'\d+%',   # Percentages
            r'according to',
            r'research shows',
        ]
        
        score = 0.0
        for pattern in factual_patterns:
            if re.search(pattern, response.lower()):
                score += 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _length_reward(self, prompt: str, response: str) -> float:
        """Configurable length reward"""
        word_count = len(response.split())
        optimal_length = 25
        deviation = abs(word_count - optimal_length)
        score = max(0.0, 1.0 - deviation / optimal_length)
        return score
    
    def _entropy_bonus(self, prompt: str, response: str) -> float:
        """Calculate entropy bonus for diverse word usage"""
        words = response.lower().split()
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * np.log2(prob + 1e-10)
        
        # Normalize entropy
        max_entropy = np.log2(len(word_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy
    
    def _apply_normalization(self, rewards: List[float]) -> List[float]:
        """Apply normalization based on configuration"""
        if not rewards or self.normalization_method == 'none':
            return rewards
        
        rewards_array = np.array(rewards)
        
        if self.normalization_method == 'standard':
            mean = np.mean(rewards_array)
            std = np.std(rewards_array)
            if std > 0:
                return ((rewards_array - mean) / std).tolist()
            return rewards
        
        return rewards
    
    def reset_components(self):
        """Reset component state"""
        self.step_count = 0
        self.component_history = []
        print("RLlama components reset")
    
    def get_last_batch_detailed_infos(self) -> Dict:
        """Return detailed information about the last batch"""
        if not self.component_history:
            return {}
        return self.component_history[-1]
    
    def get_component_analysis(self) -> Dict:
        """Get analysis of component performance over time"""
        if not self.component_history:
            return {'total_steps': 0, 'avg_rewards': {}}
        
        analysis = {
            'total_steps': len(self.component_history),
            'avg_rewards': {},
        }
        
        # Analyze component contributions
        component_trends = {}
        for history in self.component_history:
            for comp_name, scores in history['components'].items():
                if comp_name not in component_trends:
                    component_trends[comp_name] = []
                component_trends[comp_name].extend(scores)
        
        # Calculate averages
        for comp_name, scores in component_trends.items():
            analysis['avg_rewards'][comp_name] = np.mean(scores) if scores else 0.0
        
        return analysis