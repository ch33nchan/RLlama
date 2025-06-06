# rllama/integration/trl_wrapper.py

import yaml
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
import re
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TRLRllamaRewardProcessor:
    """
    Complete TRL integration for RLlama reward processing
    Handles multi-component reward calculation, scheduling, and normalization
    """
    
    def __init__(self, rllama_config_path: str, auto_register_llm_components: bool = True):
        self.config_path = rllama_config_path
        self.step_count = 0
        self.component_history = []
        self.batch_statistics = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.load_config()
        
        # Validate components
        self._validate_components()
        
        logger.info("TRLRllamaRewardProcessor initialized successfully.")
        logger.info(f"  Loaded {len(self.components)} reward components")
        logger.info(f"  Normalization method: {self.normalization_method}")
        logger.info(f"  Device: {self.device}")
        
    def load_config(self):
        """Load YAML config and parse components"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Parse composer configuration
            composer_config = self.config.get('composer', {})
            self.components = composer_config.get('components', [])
            
            # Parse shaper configuration
            shaper_config = self.config.get('shaper', {})
            self.normalization_method = shaper_config.get('normalization_method', 'none')
            self.normalization_params = shaper_config.get('params', {})
            
            # Parse optimizer configuration
            optimizer_config = self.config.get('optimizer', {})
            self.optimizer_enabled = optimizer_config.get('enabled', False)
            
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            self._load_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            self._load_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration if config file fails"""
        logger.warning("Loading default configuration")
        self.config = {}
        self.components = [
            {'type': 'CoherenceReward', 'weight': 1.0},
            {'type': 'HelpfulnessReward', 'weight': 0.8}
        ]
        self.normalization_method = 'standard'
        self.normalization_params = {}
        self.optimizer_enabled = False
    
    def _validate_components(self):
        """Validate that all components are properly configured"""
        valid_components = []
        available_types = [
            'CoherenceReward', 'HelpfulnessReward', 'DiversityReward',
            'ConcisenessReward', 'FactualityReward', 'LengthReward',
            'EntropyBonus', 'RepetitionPenalty', 'SentimentReward',
            'ReadabilityReward', 'ToxicityReward'
        ]
        
        for component in self.components:
            if 'type' not in component:
                logger.warning(f"Component missing 'type' field: {component}")
                continue
            
            if component['type'] not in available_types:
                logger.warning(f"Unknown component type: {component['type']}")
                continue
            
            # Set default weight if missing
            if 'weight' not in component:
                component['weight'] = 1.0
                logger.info(f"Set default weight 1.0 for {component['type']}")
            
            valid_components.append(component)
        
        self.components = valid_components
        logger.info(f"Validated {len(valid_components)} components")
            
    def compute_rewards(self, prompts_text: List[str], responses_text: List[str], 
                       model_specific_infos: Optional[Any] = None) -> List[float]:
        """
        Main method to compute rewards for prompt-response pairs
        
        Args:
            prompts_text: List of input prompts
            responses_text: List of model responses
            model_specific_infos: Optional additional information
            
        Returns:
            List of computed rewards (floats)
        """
        if not prompts_text or not responses_text:
            logger.warning("Empty prompts or responses provided")
            return []
        
        if len(prompts_text) != len(responses_text):
            logger.error(f"Mismatch: {len(prompts_text)} prompts, {len(responses_text)} responses")
            return [0.0] * len(prompts_text)
        
        try:
            batch_rewards = []
            component_details = {}
            
            # Initialize component tracking
            for component in self.components:
                component_details[component.get('type', 'unknown')] = []
            
            # Process each prompt-response pair
            for i, (prompt, response) in enumerate(zip(prompts_text, responses_text)):
                try:
                    total_reward = 0.0
                    sample_components = {}
                    
                    # Calculate each component reward
                    for component in self.components:
                        component_type = component.get('type', '')
                        base_weight = component.get('weight', 1.0)
                        component_params = component.get('params', {})
                        
                        # Apply scheduling if configured
                        current_weight = self._apply_scheduling(component, base_weight)
                        
                        # Calculate component-specific reward
                        component_reward = self._calculate_component_reward(
                            component_type, prompt, response, **component_params
                        )
                        
                        weighted_reward = current_weight * component_reward
                        total_reward += weighted_reward
                        
                        # Track for analysis
                        sample_components[component_type] = {
                            'raw_score': component_reward,
                            'weight': current_weight,
                            'weighted_score': weighted_reward
                        }
                        component_details[component_type].append(weighted_reward)
                    
                    batch_rewards.append(total_reward)
                    
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    batch_rewards.append(0.0)
                    # Add zero scores for tracking
                    for component in self.components:
                        component_type = component.get('type', '')
                        component_details[component_type].append(0.0)
            
            # Apply normalization
            normalized_rewards = self._apply_normalization(batch_rewards)
            
            # Store batch statistics
            batch_stats = {
                'step': self.step_count,
                'components': component_details,
                'raw_rewards': batch_rewards,
                'normalized_rewards': normalized_rewards,
                'batch_size': len(batch_rewards),
                'mean_raw_reward': np.mean(batch_rewards) if batch_rewards else 0.0,
                'std_raw_reward': np.std(batch_rewards) if batch_rewards else 0.0,
                'mean_normalized_reward': np.mean(normalized_rewards) if normalized_rewards else 0.0,
                'std_normalized_reward': np.std(normalized_rewards) if normalized_rewards else 0.0
            }
            
            self.component_history.append(batch_stats)
            self.batch_statistics.append(batch_stats)
            
            self.step_count += 1
            
            logger.info(f"Step {self.step_count}: Processed {len(batch_rewards)} samples, "
                       f"mean reward: {batch_stats['mean_raw_reward']:.3f} -> {batch_stats['mean_normalized_reward']:.3f}")
            
            return normalized_rewards
            
        except Exception as e:
            logger.error(f"Error in compute_rewards: {e}")
            return [0.0] * len(prompts_text)
    
    def _apply_scheduling(self, component: Dict, base_weight: float) -> float:
        """Apply weight scheduling based on component configuration"""
        schedule = component.get('schedule', {})
        if not schedule:
            return base_weight
        
        schedule_type = schedule.get('type', 'constant')
        
        try:
            if schedule_type == 'exponential_decay':
                decay_rate = schedule.get('decay_rate', 0.95)
                return base_weight * (decay_rate ** self.step_count)
            
            elif schedule_type == 'linear_decay':
                decay_steps = schedule.get('decay_steps', 100)
                if self.step_count >= decay_steps:
                    return 0.0
                return base_weight * (1.0 - self.step_count / decay_steps)
            
            elif schedule_type == 'curriculum':
                max_steps = schedule.get('max_steps', 100)
                return base_weight * min(1.0, self.step_count / max_steps)
            
            elif schedule_type == 'step_schedule':
                steps = schedule.get('steps', [])
                current_weight = base_weight
                for step_info in steps:
                    if self.step_count >= step_info.get('step', 0):
                        current_weight = step_info.get('weight', base_weight)
                return current_weight
            
        except Exception as e:
            logger.warning(f"Error in scheduling for {component.get('type', 'unknown')}: {e}")
        
        return base_weight
    
    def _calculate_component_reward(self, component_type: str, prompt: str, response: str, **kwargs) -> float:
        """Calculate reward for specific component types"""
        
        try:
            if component_type == "CoherenceReward":
                return self._coherence_reward(prompt, response, **kwargs)
            elif component_type == "HelpfulnessReward":
                return self._helpfulness_reward(prompt, response, **kwargs)
            elif component_type == "DiversityReward":
                return self._diversity_reward(prompt, response, **kwargs)
            elif component_type == "ConcisenessReward":
                return self._conciseness_reward(prompt, response, **kwargs)
            elif component_type == "FactualityReward":
                return self._factuality_reward(prompt, response, **kwargs)
            elif component_type == "LengthReward":
                return self._length_reward(prompt, response, **kwargs)
            elif component_type == "EntropyBonus":
                return self._entropy_bonus(prompt, response, **kwargs)
            elif component_type == "RepetitionPenalty":
                return self._repetition_penalty(prompt, response, **kwargs)
            elif component_type == "SentimentReward":
                return self._sentiment_reward(prompt, response, **kwargs)
            elif component_type == "ReadabilityReward":
                return self._readability_reward(prompt, response, **kwargs)
            elif component_type == "ToxicityReward":
                return self._toxicity_reward(prompt, response, **kwargs)
            else:
                logger.warning(f"Unknown component type: {component_type}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating {component_type}: {e}")
            return 0.0
    
    # Individual reward component implementations
    def _coherence_reward(self, prompt: str, response: str, 
                         min_sentences: int = 2, transition_bonus: float = 0.2, 
                         structure_penalty: float = 0.3, **kwargs) -> float:
        """Reward coherent, well-structured responses"""
        score = 0.0
        
        # Check for proper sentence structure
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) >= min_sentences:
            score += 0.4
        
        # Check for transitional words
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                      'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence']
        if any(trans in response.lower() for trans in transitions):
            score += transition_bonus
        
        # Check for logical connectors
        connectors = ['because', 'since', 'due to', 'as a result', 'for example', 'such as']
        if any(conn in response.lower() for conn in connectors):
            score += 0.1
        
        # Penalize very short responses
        word_count = len(response.split())
        if word_count < 5:
            score -= structure_penalty
        elif 10 <= word_count <= 50:
            score += 0.3
        
        return np.clip(score, 0.0, 1.0)
    
    def _helpfulness_reward(self, prompt: str, response: str,
                           overlap_weight: float = 0.7, question_bonus: float = 0.3, **kwargs) -> float:
        """Reward responses that address the prompt"""
        # Remove common stop words for better relevance
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        prompt_words = set(prompt.lower().split()) - stop_words
        response_words = set(response.lower().split()) - stop_words
        
        # Calculate word overlap
        if len(prompt_words) == 0:
            overlap_score = 0.0
        else:
            overlap = len(prompt_words.intersection(response_words))
            overlap_score = overlap / len(prompt_words)
        
        # Bonus for answering questions
        question_score = 0.0
        if '?' in prompt:
            answer_indicators = ['because', 'since', 'due to', 'is', 'are', 'was', 'were', 
                               'yes', 'no', 'can', 'will', 'should', 'would']
            if any(word in response.lower() for word in answer_indicators):
                question_score = question_bonus
        
        # Bonus for addressing specific request types
        request_bonus = 0.0
        if any(word in prompt.lower() for word in ['describe', 'explain', 'tell me', 'what is']):
            if len(response.split()) >= 10:  # Adequate length for explanation
                request_bonus = 0.2
        
        total_score = overlap_score * overlap_weight + question_score + request_bonus
        return np.clip(total_score, 0.0, 1.0)
    
    def _diversity_reward(self, prompt: str, response: str, **kwargs) -> float:
        """Reward lexical diversity"""
        words = response.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        diversity_ratio = unique_words / total_words
        
        # Bonus for using varied vocabulary (more than just simple words)
        complex_words = [word for word in words if len(word) > 6]
        complexity_bonus = min(0.2, len(set(complex_words)) / max(len(words), 1))
        
        return np.clip(diversity_ratio + complexity_bonus, 0.0, 1.0)
    
    def _conciseness_reward(self, prompt: str, response: str, 
                           optimal_min: int = 5, optimal_max: int = 25, **kwargs) -> float:
        """Reward concise but complete responses"""
        word_count = len(response.split())
        
        if word_count < optimal_min:
            return 0.1  # Too short
        elif optimal_min <= word_count <= optimal_max:
            return 1.0  # Optimal length
        elif optimal_max < word_count <= optimal_max * 2:
            return 0.7  # Acceptable but verbose
        else:
            return max(0.2, 1.0 - (word_count - optimal_max) / 100)  # Penalize verbosity
    
    def _factuality_reward(self, prompt: str, response: str, **kwargs) -> float:
        """Basic factuality assessment"""
        score = 0.0
        
        # Look for factual indicators
        factual_patterns = [
            r'\d{4}',  # Years
            r'\d+%',   # Percentages
            r'\d+\s*(km|miles|meters|feet|pounds|kg)',  # Measurements
            r'according to',
            r'research shows',
            r'studies indicate',
            r'data suggests',
            r'evidence shows'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, response.lower()):
                score += 0.15
        
        # Check for hedging language appropriateness
        hedge_words = ['possibly', 'might', 'could', 'may', 'perhaps', 'likely']
        factual_requests = ['fact', 'true', 'correct', 'accurate', 'exactly', 'precisely']
        
        if any(word in prompt.lower() for word in factual_requests):
            # If factual accuracy is requested, penalize too much hedging
            hedge_count = sum(1 for word in hedge_words if word in response.lower())
            if hedge_count > 2:
                score -= 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _length_reward(self, prompt: str, response: str, 
                      optimal_length: int = 25, tolerance: float = 0.3, **kwargs) -> float:
        """Configurable length reward"""
        word_count = len(response.split())
        deviation = abs(word_count - optimal_length)
        tolerance_range = optimal_length * tolerance
        
        if deviation <= tolerance_range:
            return 1.0
        else:
            score = max(0.0, 1.0 - (deviation - tolerance_range) / optimal_length)
            return score
    
    def _entropy_bonus(self, prompt: str, response: str, **kwargs) -> float:
        """Calculate entropy bonus for diverse word usage"""
        words = response.lower().split()
        if len(words) < 2:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * np.log2(prob + 1e-10)
        
        # Normalize entropy
        max_possible_entropy = np.log2(len(word_counts))
        if max_possible_entropy > 0:
            normalized_entropy = entropy / max_possible_entropy
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy
    
    def _repetition_penalty(self, prompt: str, response: str,
                           ngram_size: int = 3, penalty_weight: float = 2.0, **kwargs) -> float:
        """Penalize repetitive content"""
        words = response.lower().split()
        if len(words) < ngram_size:
            return 1.0  # No penalty for very short responses
        
        # Check for repeated n-grams
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = ' '.join(words[i:i+ngram_size])
            ngrams.append(ngram)
        
        if not ngrams:
            return 1.0
        
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
        
        penalty = repeated_ngrams / len(ngrams)
        reward = max(0.0, 1.0 - penalty * penalty_weight)
        
        return reward
    
    def _sentiment_reward(self, prompt: str, response: str, **kwargs) -> float:
        """Basic sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'positive', 'helpful', 'useful', 'beneficial', 'effective']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'wrong', 
                         'problem', 'issue', 'difficult', 'challenging', 'poor']
        
        response_lower = response.lower()
        positive_score = sum(1 for word in positive_words if word in response_lower)
        negative_score = sum(1 for word in negative_words if word in response_lower)
        
        # Context-aware sentiment
        if any(word in prompt.lower() for word in ['positive', 'good', 'benefit', 'advantage', 'praise']):
            return min(1.0, positive_score / max(positive_score + negative_score, 1) + 0.2)
        elif any(word in prompt.lower() for word in ['negative', 'problem', 'issue', 'criticism']):
            return min(1.0, negative_score / max(positive_score + negative_score, 1) + 0.2)
        
        # Neutral sentiment preference
        return 0.5 + 0.1 * (positive_score - negative_score) / max(len(response.split()), 1)
    
    def _readability_reward(self, prompt: str, response: str,
                           optimal_sentence_length: Tuple[int, int] = (10, 20), **kwargs) -> float:
        """Simple readability assessment"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        words = response.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average words per sentence
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Score based on optimal sentence length
        optimal_min, optimal_max = optimal_sentence_length
        if optimal_min <= avg_words_per_sentence <= optimal_max:
            length_score = 1.0
        else:
            if avg_words_per_sentence < optimal_min:
                deviation = optimal_min - avg_words_per_sentence
            else:
                deviation = avg_words_per_sentence - optimal_max
            length_score = max(0.0, 1.0 - deviation / optimal_max)
        
        # Bonus for varied sentence lengths
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            length_variance = np.std(sentence_lengths) / np.mean(sentence_lengths) if sentence_lengths else 0
            variance_bonus = min(0.2, length_variance)
        else:
            variance_bonus = 0.0
        
        return min(1.0, length_score + variance_bonus)
    
    def _toxicity_reward(self, prompt: str, response: str, **kwargs) -> float:
        """Basic toxicity detection (placeholder for more sophisticated models)"""
        toxic_indicators = [
            'hate', 'stupid', 'idiot', 'kill', 'die', 'murder', 'violence',
            'racist', 'sexist', 'offensive', 'inappropriate', 'vulgar'
        ]
        
        response_lower = response.lower()
        toxic_count = sum(1 for word in toxic_indicators if word in response_lower)
        
        # Simple toxicity penalty
        if toxic_count == 0:
            return 1.0
        else:
            penalty = min(1.0, toxic_count * 0.5)
            return max(0.0, 1.0 - penalty)
    
    def _apply_normalization(self, rewards: List[float]) -> List[float]:
        """Apply normalization based on configuration"""
        if not rewards or self.normalization_method == 'none':
            return rewards
        
        rewards_array = np.array(rewards)
        
        try:
            if self.normalization_method == 'standard':
                # Z-score normalization
                mean = np.mean(rewards_array)
                std = np.std(rewards_array)
                if std > 1e-8:  # Avoid division by zero
                    return ((rewards_array - mean) / std).tolist()
                return rewards
            
            elif self.normalization_method == 'minmax':
                # Min-max normalization
                min_val = np.min(rewards_array)
                max_val = np.max(rewards_array)
                if max_val > min_val:
                    return ((rewards_array - min_val) / (max_val - min_val)).tolist()
                return rewards
            
            elif self.normalization_method == 'robust':
                # Robust normalization using median and IQR
                median = np.median(rewards_array)
                q75, q25 = np.percentile(rewards_array, [75, 25])
                iqr = q75 - q25
                if iqr > 1e-8:
                    return ((rewards_array - median) / iqr).tolist()
                return rewards
            
            elif self.normalization_method == 'sigmoid':
                # Sigmoid normalization
                return (1 / (1 + np.exp(-rewards_array))).tolist()
            
        except Exception as e:
            logger.error(f"Error in normalization: {e}")
        
        return rewards
    
    def reset_components(self):
        """Reset component state (useful for periodic resets)"""
        self.step_count = 0
        self.component_history = []
        self.batch_statistics = []
        logger.info("RLlama components reset")
    
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
            'component_trends': {},
            'avg_rewards': {},
            'reward_statistics': {}
        }
        
        # Analyze component contributions
        all_components = {}
        for history in self.component_history:
            for comp_name, scores in history['components'].items():
                if comp_name not in all_components:
                    all_components[comp_name] = []
                all_components[comp_name].extend(scores)
        
        # Calculate statistics for each component
        for comp_name, scores in all_components.items():
            if scores:
                analysis['avg_rewards'][comp_name] = float(np.mean(scores))
                analysis['component_trends'][comp_name] = scores
                analysis['reward_statistics'][comp_name] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'count': len(scores)
                }
        
        # Overall statistics
        if self.batch_statistics:
            raw_rewards = []
            normalized_rewards = []
            for batch in self.batch_statistics:
                raw_rewards.extend(batch['raw_rewards'])
                normalized_rewards.extend(batch['normalized_rewards'])
            
            analysis['overall_statistics'] = {
                'raw_rewards': {
                    'mean': float(np.mean(raw_rewards)) if raw_rewards else 0.0,
                    'std': float(np.std(raw_rewards)) if raw_rewards else 0.0,
                    'min': float(np.min(raw_rewards)) if raw_rewards else 0.0,
                    'max': float(np.max(raw_rewards)) if raw_rewards else 0.0
                },
                'normalized_rewards': {
                    'mean': float(np.mean(normalized_rewards)) if normalized_rewards else 0.0,
                    'std': float(np.std(normalized_rewards)) if normalized_rewards else 0.0,
                    'min': float(np.min(normalized_rewards)) if normalized_rewards else 0.0,
                    'max': float(np.max(normalized_rewards)) if normalized_rewards else 0.0
                }
            }
        
        return analysis
    
    def save_analysis(self, filepath: str):
        """Save detailed analysis to file"""
        analysis = self.get_component_analysis()
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
    
    def get_reward_trends(self) -> Dict[str, List[float]]:
        """Get reward trends over time for plotting"""
        trends = {
            'raw_rewards': [],
            'normalized_rewards': [],
            'step_numbers': []
        }
        
        for i, batch in enumerate(self.batch_statistics):
            trends['raw_rewards'].append(batch['mean_raw_reward'])
            trends['normalized_rewards'].append(batch['mean_normalized_reward'])
            trends['step_numbers'].append(i)
        
        return trends