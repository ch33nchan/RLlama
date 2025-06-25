#!/usr/bin/env python3
"""
Length-specific reward components for text generation and sequence modeling.
These components provide sophisticated length-based reward functions for various use cases.
"""

import numpy as np
import math
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import defaultdict

from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class TokenLengthReward(BaseReward):
    """
    Reward component based on token count in text.
    Provides more accurate length measurement for language models.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 target_tokens: int = 50,
                 tokenizer_key: str = "tokenizer",
                 penalty_type: str = "quadratic",
                 strength: float = 0.01,
                 min_tokens: int = 5,
                 max_tokens: int = 500,
                 **kwargs):
        """
        Initialize token length reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            target_tokens: Target number of tokens
            tokenizer_key: Key in context containing tokenizer
            penalty_type: Type of penalty ("linear", "quadratic", "exponential")
            strength: Scaling factor for the reward
            min_tokens: Minimum acceptable token count
            max_tokens: Maximum acceptable token count
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.target_tokens = target_tokens
        self.tokenizer_key = tokenizer_key
        self.penalty_type = penalty_type
        self.strength = strength
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate token-based length reward."""
        text = context.get(self.text_key, "")
        if not text or not isinstance(text, str):
            return 0.0
            
        # Get token count
        token_count = self._count_tokens(text, context)
        
        # Calculate deviation from target
        if token_count < self.min_tokens:
            deviation = self.min_tokens - token_count
            penalty_multiplier = 2.0  # Extra penalty for too short
        elif token_count > self.max_tokens:
            deviation = token_count - self.max_tokens
            penalty_multiplier = 1.5  # Penalty for too long
        else:
            deviation = abs(token_count - self.target_tokens)
            penalty_multiplier = 1.0
            
        # Apply penalty function
        if self.penalty_type == "linear":
            penalty = deviation
        elif self.penalty_type == "quadratic":
            penalty = deviation ** 2
        elif self.penalty_type == "exponential":
            penalty = math.exp(deviation / self.target_tokens) - 1
        else:
            penalty = deviation
            
        # Calculate final reward
        reward = -self.strength * penalty * penalty_multiplier
        
        return reward
        
    def _count_tokens(self, text: str, context: Dict[str, Any]) -> int:
        """Count tokens in text using tokenizer if available."""
        tokenizer = context.get(self.tokenizer_key)
        
        if tokenizer is not None:
            try:
                # Try different tokenizer interfaces
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(text)
                    return len(tokens)
                elif hasattr(tokenizer, 'tokenize'):
                    tokens = tokenizer.tokenize(text)
                    return len(tokens)
                elif callable(tokenizer):
                    tokens = tokenizer(text)
                    if hasattr(tokens, 'input_ids'):
                        return len(tokens.input_ids)
                    elif isinstance(tokens, list):
                        return len(tokens)
            except Exception:
                pass
                
        # Fallback to simple word-based tokenization
        return len(text.split())

@register_reward_component
class SentenceLengthReward(BaseReward):
    """
    Reward component based on sentence count and average sentence length.
    Encourages well-structured text with appropriate sentence boundaries.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 target_sentences: int = 3,
                 target_avg_length: int = 20,
                 sentence_weight: float = 0.6,
                 avg_length_weight: float = 0.4,
                 strength: float = 0.01,
                 **kwargs):
        """
        Initialize sentence length reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            target_sentences: Target number of sentences
            target_avg_length: Target average sentence length (in words)
            sentence_weight: Weight for sentence count component
            avg_length_weight: Weight for average length component
            strength: Overall scaling factor
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.target_sentences = target_sentences
        self.target_avg_length = target_avg_length
        self.sentence_weight = sentence_weight
        self.avg_length_weight = avg_length_weight
        self.strength = strength
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate sentence-based reward."""
        text = context.get(self.text_key, "")
        if not text or not isinstance(text, str):
            return 0.0
            
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return -self.strength
            
        # Calculate sentence count deviation
        sentence_count = len(sentences)
        sentence_deviation = abs(sentence_count - self.target_sentences)
        
        # Calculate average sentence length
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_length = np.mean(sentence_lengths)
        length_deviation = abs(avg_length - self.target_avg_length)
        
        # Calculate component rewards
        sentence_reward = -self.sentence_weight * sentence_deviation ** 2
        length_reward = -self.avg_length_weight * length_deviation ** 2
        
        # Combine rewards
        total_reward = self.strength * (sentence_reward + length_reward)
        
        return total_reward
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex to split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

@register_reward_component
class ParagraphLengthReward(BaseReward):
    """
    Reward component based on paragraph structure and length.
    Encourages well-organized text with appropriate paragraph breaks.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 target_paragraphs: int = 2,
                 target_sentences_per_paragraph: int = 3,
                 min_paragraph_length: int = 50,
                 max_paragraph_length: int = 200,
                 structure_weight: float = 0.5,
                 length_weight: float = 0.5,
                 strength: float = 0.01,
                 **kwargs):
        """
        Initialize paragraph length reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            target_paragraphs: Target number of paragraphs
            target_sentences_per_paragraph: Target sentences per paragraph
            min_paragraph_length: Minimum paragraph length (characters)
            max_paragraph_length: Maximum paragraph length (characters)
            structure_weight: Weight for paragraph structure component
            length_weight: Weight for paragraph length component
            strength: Overall scaling factor
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.target_paragraphs = target_paragraphs
        self.target_sentences_per_paragraph = target_sentences_per_paragraph
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        self.structure_weight = structure_weight
        self.length_weight = length_weight
        self.strength = strength
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate paragraph-based reward."""
        text = context.get(self.text_key, "")
        if not text or not isinstance(text, str):
            return 0.0
            
        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)
        
        if not paragraphs:
            return -self.strength
            
        # Calculate structure reward
        structure_reward = self._calculate_structure_reward(paragraphs)
        
        # Calculate length reward
        length_reward = self._calculate_length_reward(paragraphs)
        
        # Combine rewards
        total_reward = self.strength * (
            self.structure_weight * structure_reward +
            self.length_weight * length_reward
        )
        
        return total_reward
        
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or explicit paragraph markers
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        # Clean up paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
        
    def _calculate_structure_reward(self, paragraphs: List[str]) -> float:
        """Calculate reward based on paragraph structure."""
        paragraph_count = len(paragraphs)
        
        # Penalty for wrong number of paragraphs
        count_penalty = abs(paragraph_count - self.target_paragraphs) ** 2
        
        # Calculate sentences per paragraph
        sentences_per_paragraph = []
        for paragraph in paragraphs:
            sentences = re.split(r'[.!?]+', paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]
            sentences_per_paragraph.append(len(sentences))
            
        # Penalty for wrong sentence distribution
        if sentences_per_paragraph:
            avg_sentences = np.mean(sentences_per_paragraph)
            sentence_penalty = abs(avg_sentences - self.target_sentences_per_paragraph) ** 2
        else:
            sentence_penalty = self.target_sentences_per_paragraph ** 2
            
        # Combine penalties
        structure_reward = -(count_penalty + sentence_penalty)
        
        return structure_reward
        
    def _calculate_length_reward(self, paragraphs: List[str]) -> float:
        """Calculate reward based on paragraph lengths."""
        length_penalties = []
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            if paragraph_length < self.min_paragraph_length:
                penalty = (self.min_paragraph_length - paragraph_length) ** 2
            elif paragraph_length > self.max_paragraph_length:
                penalty = (paragraph_length - self.max_paragraph_length) ** 2
            else:
                penalty = 0
                
            length_penalties.append(penalty)
            
        # Average penalty across paragraphs
        avg_penalty = np.mean(length_penalties) if length_penalties else 0
        
        return -avg_penalty

@register_reward_component
class OptimalLengthReward(BaseReward):
    """
    Advanced reward component that adapts to find optimal length based on content quality.
    Uses dynamic programming to balance length with content density.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 quality_key: str = "quality_score",
                 min_length: int = 10,
                 max_length: int = 1000,
                 density_weight: float = 0.6,
                 length_weight: float = 0.4,
                 adaptation_rate: float = 0.01,
                 **kwargs):
        """
        Initialize optimal length reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            quality_key: Key in context containing quality score
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            density_weight: Weight for content density
            length_weight: Weight for length optimization
            adaptation_rate: Rate of adaptation for optimal length
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.quality_key = quality_key
        self.min_length = min_length
        self.max_length = max_length
        self.density_weight = density_weight
        self.length_weight = length_weight
        self.adaptation_rate = adaptation_rate
        
        # Track optimal length estimates
        self.length_quality_history = []
        self.estimated_optimal_length = (min_length + max_length) / 2
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate optimal length reward."""
        text = context.get(self.text_key, "")
        if not text or not isinstance(text, str):
            return 0.0
            
        # Get text length
        text_length = len(text)
        
        # Get quality score if available
        quality_score = context.get(self.quality_key, 0.5)
        if not isinstance(quality_score, (int, float)):
            quality_score = 0.5
            
        # Update history
        self.length_quality_history.append((text_length, quality_score))
        if len(self.length_quality_history) > 100:
            self.length_quality_history = self.length_quality_history[-100:]
            
        # Update optimal length estimate
        self._update_optimal_length()
        
        # Calculate content density
        density = self._calculate_content_density(text, quality_score)
        
        # Calculate length optimality
        length_optimality = self._calculate_length_optimality(text_length)
        
        # Combine rewards
        total_reward = (
            self.density_weight * density +
            self.length_weight * length_optimality
        )
        
        return total_reward
        
    def _calculate_content_density(self, text: str, quality_score: float) -> float:
        """Calculate content density (quality per unit length)."""
        text_length = len(text)
        
        if text_length == 0:
            return 0.0
            
        # Basic density calculation
        density = quality_score / math.log(text_length + 1)
        
        # Normalize density
        normalized_density = 2.0 * (density - 0.5)  # Scale to [-1, 1]
        
        return normalized_density
        
    def _calculate_length_optimality(self, text_length: int) -> float:
        """Calculate how close the length is to optimal."""
        # Penalty for being outside acceptable range
        if text_length < self.min_length:
            return -2.0 * (self.min_length - text_length) / self.min_length
        elif text_length > self.max_length:
            return -2.0 * (text_length - self.max_length) / self.max_length
            
        # Reward for being close to estimated optimal length
        deviation = abs(text_length - self.estimated_optimal_length)
        max_deviation = max(
            abs(self.min_length - self.estimated_optimal_length),
            abs(self.max_length - self.estimated_optimal_length)
        )
        
        if max_deviation > 0:
            optimality = 1.0 - (deviation / max_deviation)
        else:
            optimality = 1.0
            
        return optimality
        
    def _update_optimal_length(self) -> None:
        """Update the estimated optimal length based on history."""
        if len(self.length_quality_history) < 10:
            return
            
        # Find length that maximizes quality
        best_quality = -float('inf')
        best_length = self.estimated_optimal_length
        
        # Group by length ranges and find average quality
        length_ranges = defaultdict(list)
        
        for length, quality in self.length_quality_history:
            # Group into ranges of 20 characters
            range_key = (length // 20) * 20
            length_ranges[range_key].append(quality)
            
        # Find range with highest average quality
        for range_start, qualities in length_ranges.items():
            avg_quality = np.mean(qualities)
            if avg_quality > best_quality:
                best_quality = avg_quality
                best_length = range_start + 10  # Middle of range
                
        # Update estimate with adaptation rate
        self.estimated_optimal_length = (
            (1 - self.adaptation_rate) * self.estimated_optimal_length +
            self.adaptation_rate * best_length
        )
        
        # Keep within bounds
        self.estimated_optimal_length = max(
            self.min_length,
            min(self.max_length, self.estimated_optimal_length)
        )
        
    def get_optimal_length_estimate(self) -> float:
        """Get current optimal length estimate."""
        return self.estimated_optimal_length
        
    def get_length_quality_stats(self) -> Dict[str, float]:
        """Get statistics about length-quality relationship."""
        if not self.length_quality_history:
            return {}
            
        lengths = [item[0] for item in self.length_quality_history]
        qualities = [item[1] for item in self.length_quality_history]
        
        # Calculate correlation
        if len(lengths) > 1:
            correlation = np.corrcoef(lengths, qualities)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
            
        return {
            'optimal_length_estimate': self.estimated_optimal_length,
            'length_quality_correlation': correlation,
            'avg_length': np.mean(lengths),
            'avg_quality': np.mean(qualities),
            'samples': len(self.length_quality_history)
        }
