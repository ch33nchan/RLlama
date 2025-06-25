#!/usr/bin/env python3
"""
LLM-specific reward components for language model training and evaluation.
These components implement sophisticated reward functions for text generation tasks.
"""

import numpy as np
import math
import re
from typing import Dict, Any, Optional, Union, List, Tuple
from collections import defaultdict, Counter
import warnings

from ..base import BaseReward
from ..registry import register_reward_component

@register_reward_component
class PerplexityReward(BaseReward):
    """
    Reward component based on perplexity of generated text.
    Lower perplexity indicates more fluent and coherent text.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 model_key: str = "language_model",
                 target_perplexity: float = 50.0,
                 scaling_factor: float = -0.01,
                 normalize: bool = True,
                 use_log_perplexity: bool = True,
                 **kwargs):
        """
        Initialize perplexity reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            model_key: Key in context containing language model for perplexity calculation
            target_perplexity: Target perplexity value (lower is better)
            scaling_factor: Factor to scale perplexity into reward
            normalize: Whether to normalize perplexity
            use_log_perplexity: Whether to use log perplexity for more stable rewards
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.model_key = model_key
        self.target_perplexity = target_perplexity
        self.scaling_factor = scaling_factor
        self.normalize = normalize
        self.use_log_perplexity = use_log_perplexity
        
        # Track perplexity statistics for normalization
        self.perplexity_history = []
        self.running_mean = target_perplexity
        self.running_std = 10.0
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate perplexity-based reward."""
        text = context.get(self.text_key, "")
        if not text or not isinstance(text, str):
            return 0.0
            
        # Calculate perplexity
        perplexity = self._calculate_perplexity(text, context)
        
        if perplexity is None or perplexity <= 0:
            return 0.0
            
        # Update statistics
        self._update_statistics(perplexity)
        
        # Apply log transformation if requested
        if self.use_log_perplexity:
            perplexity = math.log(perplexity + 1e-8)
            target = math.log(self.target_perplexity + 1e-8)
        else:
            target = self.target_perplexity
            
        # Normalize if requested
        if self.normalize and len(self.perplexity_history) > 10:
            perplexity = (perplexity - self.running_mean) / (self.running_std + 1e-8)
            target = (target - self.running_mean) / (self.running_std + 1e-8)
            
        # Calculate reward (negative distance from target)
        reward = self.scaling_factor * abs(perplexity - target)
        
        return reward
        
    def _calculate_perplexity(self, text: str, context: Dict[str, Any]) -> Optional[float]:
        """Calculate perplexity of text."""
        # Check if language model is provided in context
        model = context.get(self.model_key)
        
        if model is not None:
            # Use provided model to calculate perplexity
            try:
                return self._model_perplexity(text, model)
            except Exception:
                pass
                
        # Fallback to simple n-gram based perplexity estimation
        return self._ngram_perplexity(text)
        
    def _model_perplexity(self, text: str, model) -> float:
        """Calculate perplexity using a language model."""
        # This is a placeholder - in practice, you'd use the actual model
        # For example, with transformers:
        # import torch
        # from transformers import GPT2LMHeadModel, GPT2Tokenizer
        # 
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # model = GPT2LMHeadModel.from_pretrained('gpt2')
        # 
        # inputs = tokenizer(text, return_tensors='pt')
        # with torch.no_grad():
        #     outputs = model(**inputs, labels=inputs['input_ids'])
        #     loss = outputs.loss
        #     perplexity = torch.exp(loss).item()
        
        # Simplified implementation
        if hasattr(model, 'calculate_perplexity'):
            return model.calculate_perplexity(text)
        else:
            # Fallback to n-gram estimation
            return self._ngram_perplexity(text)
            
    def _ngram_perplexity(self, text: str, n: int = 3) -> float:
        """Estimate perplexity using n-gram statistics."""
        words = text.lower().split()
        
        if len(words) < n:
            return 100.0  # High perplexity for very short text
            
        # Create n-grams
        ngrams = []
        context_counts = defaultdict(int)
        ngram_counts = defaultdict(int)
        
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            context = ngram[:-1]
            
            ngrams.append(ngram)
            ngram_counts[ngram] += 1
            context_counts[context] += 1
            
        # Calculate log probability
        log_prob = 0.0
        for ngram in ngrams:
            context = ngram[:-1]
            
            # Smoothed probability (add-1 smoothing)
            prob = (ngram_counts[ngram] + 1) / (context_counts[context] + len(set(words)))
            log_prob += math.log(prob)
            
        # Calculate perplexity
        avg_log_prob = log_prob / len(ngrams)
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
        
    def _update_statistics(self, perplexity: float) -> None:
        """Update running statistics for normalization."""
        self.perplexity_history.append(perplexity)
        
        # Keep only recent history
        if len(self.perplexity_history) > 1000:
            self.perplexity_history = self.perplexity_history[-1000:]
            
        # Update running statistics
        if len(self.perplexity_history) > 1:
            self.running_mean = np.mean(self.perplexity_history)
            self.running_std = np.std(self.perplexity_history) + 1e-8

@register_reward_component
class SemanticSimilarityReward(BaseReward):
    """
    Reward component based on semantic similarity between query and response.
    Uses various similarity metrics including embedding-based approaches.
    """
    
    def __init__(self,
                 query_key: str = "query",
                 response_key: str = "response",
                 similarity_metric: str = "cosine",
                 embedding_model_key: str = "embedding_model",
                 target_similarity: float = 0.7,
                 scaling_factor: float = 1.0,
                 **kwargs):
        """
        Initialize semantic similarity reward component.
        
        Args:
            query_key: Key in context containing query/prompt
            response_key: Key in context containing response
            similarity_metric: Similarity metric ("cosine", "jaccard", "bleu", "rouge")
            embedding_model_key: Key in context containing embedding model
            target_similarity: Target similarity score
            scaling_factor: Factor to scale similarity into reward
        """
        super().__init__(**kwargs)
        self.query_key = query_key
        self.response_key = response_key
        self.similarity_metric = similarity_metric
        self.embedding_model_key = embedding_model_key
        self.target_similarity = target_similarity
        self.scaling_factor = scaling_factor
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate semantic similarity reward."""
        query = context.get(self.query_key, "")
        response = context.get(self.response_key, "")
        
        if not query or not response:
            return 0.0
            
        # Calculate similarity based on metric
        if self.similarity_metric == "cosine":
            similarity = self._cosine_similarity(query, response, context)
        elif self.similarity_metric == "jaccard":
            similarity = self._jaccard_similarity(query, response)
        elif self.similarity_metric == "bleu":
            similarity = self._bleu_similarity(query, response)
        elif self.similarity_metric == "rouge":
            similarity = self._rouge_similarity(query, response)
        else:
            similarity = self._cosine_similarity(query, response, context)
            
        # Scale similarity to reward
        reward = self.scaling_factor * similarity
        
        return reward
        
    def _cosine_similarity(self, query: str, response: str, context: Dict[str, Any]) -> float:
        """Calculate cosine similarity using embeddings."""
        embedding_model = context.get(self.embedding_model_key)
        
        if embedding_model is not None:
            try:
                # Use provided embedding model
                query_embedding = self._get_embedding(query, embedding_model)
                response_embedding = self._get_embedding(response, embedding_model)
                
                if query_embedding is not None and response_embedding is not None:
                    return self._cosine_sim_vectors(query_embedding, response_embedding)
            except Exception:
                pass
                
        # Fallback to TF-IDF based similarity
        return self._tfidf_cosine_similarity(query, response)
        
    def _get_embedding(self, text: str, model) -> Optional[np.ndarray]:
        """Get embedding for text using provided model."""
        if hasattr(model, 'encode'):
            return model.encode(text)
        elif hasattr(model, 'get_embedding'):
            return model.get_embedding(text)
        else:
            return None
            
    def _cosine_sim_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _tfidf_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based cosine similarity."""
        # Tokenize
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Get vocabulary
        vocab = set(words1 + words2)
        
        if not vocab:
            return 0.0
            
        # Calculate TF-IDF vectors
        vec1 = self._tfidf_vector(words1, vocab)
        vec2 = self._tfidf_vector(words2, vocab)
        
        return self._cosine_sim_vectors(vec1, vec2)
        
    def _tfidf_vector(self, words: List[str], vocab: set) -> np.ndarray:
        """Calculate TF-IDF vector for words."""
        word_counts = Counter(words)
        total_words = len(words)
        
        vector = []
        for word in sorted(vocab):
            tf = word_counts[word] / total_words if total_words > 0 else 0
            # Simplified IDF (assuming single document)
            idf = 1.0
            vector.append(tf * idf)
            
        return np.array(vector)
        
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
        
    def _bleu_similarity(self, reference: str, candidate: str) -> float:
        """Calculate simplified BLEU score."""
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if not cand_words:
            return 0.0
            
        # Calculate 1-gram precision
        ref_counts = Counter(ref_words)
        cand_counts = Counter(cand_words)
        
        matches = 0
        for word, count in cand_counts.items():
            matches += min(count, ref_counts.get(word, 0))
            
        precision = matches / len(cand_words)
        
        # Brevity penalty
        bp = min(1.0, len(cand_words) / max(1, len(ref_words)))
        
        return bp * precision
        
    def _rouge_similarity(self, reference: str, candidate: str) -> float:
        """Calculate simplified ROUGE-1 score."""
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        
        if not ref_words:
            return 0.0
            
        overlap = ref_words.intersection(cand_words)
        return len(overlap) / len(ref_words)

@register_reward_component
class ToxicityReward(BaseReward):
    """
    Reward component that penalizes toxic or harmful content.
    Uses various toxicity detection methods.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 toxicity_model_key: str = "toxicity_model",
                 penalty_strength: float = -5.0,
                 toxicity_threshold: float = 0.5,
                 use_keyword_filter: bool = True,
                 **kwargs):
        """
        Initialize toxicity reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            toxicity_model_key: Key in context containing toxicity detection model
            penalty_strength: Penalty for toxic content (should be negative)
            toxicity_threshold: Threshold for toxicity detection
            use_keyword_filter: Whether to use keyword-based filtering as fallback
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.toxicity_model_key = toxicity_model_key
        self.penalty_strength = penalty_strength
        self.toxicity_threshold = toxicity_threshold
        self.use_keyword_filter = use_keyword_filter
        
        # Toxic keywords for fallback detection
        self.toxic_keywords = [
            "hate", "kill", "murder", "violence", "attack", "harm", "hurt",
            "racist", "sexist", "discrimination", "offensive", "insult",
            "threat", "abuse", "harass", "bully", "toxic", "poison"
        ]
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate toxicity-based reward (penalty)."""
        text = context.get(self.text_key, "")
        if not text or not isinstance(text, str):
            return 0.0
            
        # Calculate toxicity score
        toxicity_score = self._calculate_toxicity(text, context)
        
        # Apply penalty if above threshold
        if toxicity_score > self.toxicity_threshold:
            penalty = self.penalty_strength * (toxicity_score - self.toxicity_threshold)
            return penalty
        else:
            return 0.0  # No penalty for non-toxic content
            
    def _calculate_toxicity(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate toxicity score for text."""
        # Try to use provided toxicity model
        toxicity_model = context.get(self.toxicity_model_key)
        
        if toxicity_model is not None:
            try:
                return self._model_toxicity(text, toxicity_model)
            except Exception:
                pass
                
        # Fallback to keyword-based detection
        if self.use_keyword_filter:
            return self._keyword_toxicity(text)
        else:
            return 0.0
            
    def _model_toxicity(self, text: str, model) -> float:
        """Calculate toxicity using a toxicity detection model."""
        if hasattr(model, 'predict_toxicity'):
            return model.predict_toxicity(text)
        elif hasattr(model, 'predict'):
            result = model.predict(text)
            if isinstance(result, dict) and 'toxicity' in result:
                return result['toxicity']
            elif isinstance(result, (int, float)):
                return float(result)
        
        return 0.0
        
    def _keyword_toxicity(self, text: str) -> float:
        """Calculate toxicity based on keyword presence."""
        text_lower = text.lower()
        
        # Count toxic keywords
        toxic_count = 0
        for keyword in self.toxic_keywords:
            toxic_count += text_lower.count(keyword)
            
        # Calculate toxicity score based on keyword density
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        toxicity_density = toxic_count / word_count
        
        # Scale to 0-1 range
        toxicity_score = min(1.0, toxicity_density * 5.0)  # Scale factor
        
        return toxicity_score

@register_reward_component
class FactualityReward(BaseReward):
    """
    Reward component that encourages factually accurate content.
    Uses various fact-checking approaches.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 query_key: str = "query",
                 fact_checker_key: str = "fact_checker",
                 knowledge_base_key: str = "knowledge_base",
                 factuality_weight: float = 1.0,
                 uncertainty_penalty: float = 0.1,
                 **kwargs):
        """
        Initialize factuality reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            query_key: Key in context containing query for context
            fact_checker_key: Key in context containing fact-checking model
            knowledge_base_key: Key in context containing knowledge base
            factuality_weight: Weight for factuality score
            uncertainty_penalty: Penalty for uncertain/ambiguous statements
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.query_key = query_key
        self.fact_checker_key = fact_checker_key
        self.knowledge_base_key = knowledge_base_key
        self.factuality_weight = factuality_weight
        self.uncertainty_penalty = uncertainty_penalty
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate factuality-based reward."""
        text = context.get(self.text_key, "")
        query = context.get(self.query_key, "")
        
        if not text:
            return 0.0
            
        # Calculate factuality score
        factuality_score = self._calculate_factuality(text, query, context)
        
        # Calculate uncertainty penalty
        uncertainty_score = self._calculate_uncertainty(text)
        
        # Combine scores
        reward = (self.factuality_weight * factuality_score - 
                 self.uncertainty_penalty * uncertainty_score)
        
        return reward
        
    def _calculate_factuality(self, text: str, query: str, context: Dict[str, Any]) -> float:
        """Calculate factuality score for text."""
        # Try to use provided fact checker
        fact_checker = context.get(self.fact_checker_key)
        
        if fact_checker is not None:
            try:
                return self._model_factuality(text, fact_checker)
            except Exception:
                pass
                
        # Try to use knowledge base
        knowledge_base = context.get(self.knowledge_base_key)
        
        if knowledge_base is not None:
            try:
                return self._knowledge_base_factuality(text, knowledge_base)
            except Exception:
                pass
                
        # Fallback to heuristic-based factuality
        return self._heuristic_factuality(text)
        
    def _model_factuality(self, text: str, fact_checker) -> float:
        """Calculate factuality using a fact-checking model."""
        if hasattr(fact_checker, 'check_facts'):
            return fact_checker.check_facts(text)
        elif hasattr(fact_checker, 'predict'):
            result = fact_checker.predict(text)
            if isinstance(result, dict) and 'factuality' in result:
                return result['factuality']
            elif isinstance(result, (int, float)):
                return float(result)
                
        return 0.5  # Neutral score
        
    def _knowledge_base_factuality(self, text: str, knowledge_base) -> float:
        """Calculate factuality using a knowledge base."""
        # Extract claims from text
        claims = self._extract_claims(text)
        
        if not claims:
            return 0.5
            
        # Check each claim against knowledge base
        factual_claims = 0
        total_claims = len(claims)
        
        for claim in claims:
            if self._verify_claim(claim, knowledge_base):
                factual_claims += 1
                
        return factual_claims / total_claims
        
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Filter for sentences that look like factual claims
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and self._looks_like_claim(sentence):
                claims.append(sentence)
                
        return claims
        
    def _looks_like_claim(self, sentence: str) -> bool:
        """Check if sentence looks like a factual claim."""
        # Simple heuristics
        factual_indicators = [
            "is", "are", "was", "were", "has", "have", "had",
            "will", "would", "can", "could", "should", "must",
            "according to", "research shows", "studies indicate"
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
        
    def _verify_claim(self, claim: str, knowledge_base) -> bool:
        """Verify a claim against knowledge base."""
        if hasattr(knowledge_base, 'verify'):
            return knowledge_base.verify(claim)
        elif hasattr(knowledge_base, 'search'):
            results = knowledge_base.search(claim)
            return len(results) > 0
        else:
            return False
            
    def _heuristic_factuality(self, text: str) -> float:
        """Calculate factuality using heuristics."""
        # Simple heuristics for factuality
        score = 0.5  # Start with neutral
        
        # Penalty for hedging language (indicates uncertainty)
        hedging_words = ["maybe", "perhaps", "possibly", "might", "could be", "seems"]
        hedging_count = sum(1 for word in hedging_words if word in text.lower())
        score -= hedging_count * 0.05
        
        # Bonus for specific numbers and dates (indicates precision)
        numbers = re.findall(r'\d+', text)
        score += min(0.2, len(numbers) * 0.02)
        
        # Bonus for citations or references
        citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.', r'according to']
        citations = sum(1 for pattern in citation_patterns 
                       if re.search(pattern, text))
        score += min(0.2, citations * 0.05)
        
        return max(0.0, min(1.0, score))
        
    def _calculate_uncertainty(self, text: str) -> float:
        """Calculate uncertainty score for text."""
        uncertainty_words = [
            "uncertain", "unclear", "ambiguous", "confusing", "vague",
            "maybe", "perhaps", "possibly", "might", "could be",
            "not sure", "don't know", "unclear", "uncertain"
        ]
        
        text_lower = text.lower()
        uncertainty_count = sum(1 for word in uncertainty_words 
                              if word in text_lower)
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
            
        uncertainty_score = uncertainty_count / word_count
        return min(1.0, uncertainty_score * 5.0)  # Scale factor

@register_reward_component
class CreativityReward(BaseReward):
    """
    Reward component that encourages creative and novel content.
    Uses various creativity metrics.
    """
    
    def __init__(self,
                 text_key: str = "response",
                 novelty_weight: float = 0.4,
                 diversity_weight: float = 0.3,
                 originality_weight: float = 0.3,
                 reference_corpus_key: str = "reference_corpus",
                 **kwargs):
        """
        Initialize creativity reward component.
        
        Args:
            text_key: Key in context containing text to evaluate
            novelty_weight: Weight for novelty score
            diversity_weight: Weight for diversity score
            originality_weight: Weight for originality score
            reference_corpus_key: Key in context containing reference corpus
        """
        super().__init__(**kwargs)
        self.text_key = text_key
        self.novelty_weight = novelty_weight
        self.diversity_weight = diversity_weight
        self.originality_weight = originality_weight
        self.reference_corpus_key = reference_corpus_key
        
        # Track generated text for novelty comparison
        self.generated_texts = []
        
    def calculate(self, context: Dict[str, Any]) -> float:
        """Calculate creativity-based reward."""
        text = context.get(self.text_key, "")
        if not text:
            return 0.0
            
        # Calculate different creativity metrics
        novelty_score = self._calculate_novelty(text, context)
        diversity_score = self._calculate_diversity(text)
        originality_score = self._calculate_originality(text, context)
        
        # Combine scores
        creativity_score = (
            self.novelty_weight * novelty_score +
            self.diversity_weight * diversity_score +
            self.originality_weight * originality_score
        )
        
        # Store text for future novelty comparisons
        self.generated_texts.append(text)
        if len(self.generated_texts) > 1000:
            self.generated_texts = self.generated_texts[-1000:]
            
        return creativity_score
        
    def _calculate_novelty(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate novelty score compared to previous generations."""
        if not self.generated_texts:
            return 1.0  # First generation is novel
            
        # Calculate similarity to previous texts
        similarities = []
        for prev_text in self.generated_texts[-100:]:  # Check last 100
            similarity = self._text_similarity(text, prev_text)
            similarities.append(similarity)
            
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0.0
        novelty = 1.0 - max_similarity
        
        return novelty
        
    def _calculate_diversity(self, text: str) -> float:
        """Calculate lexical diversity within the text."""
        words = text.lower().split()
        
        if len(words) < 2:
            return 0.0
            
        # Type-token ratio (unique words / total words)
        unique_words = len(set(words))
        total_words = len(words)
        
        diversity = unique_words / total_words
        
        # Bonus for varied sentence structures
        sentences = re.split(r'[.!?]+', text)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            structure_bonus = min(0.2, length_variance / 100)  # Normalize
            diversity += structure_bonus
            
        return min(1.0, diversity)
        
    def _calculate_originality(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate originality compared to reference corpus."""
        reference_corpus = context.get(self.reference_corpus_key)
        
        if reference_corpus is None:
            # Fallback to simple originality metrics
            return self._heuristic_originality(text)
            
        # Calculate similarity to reference corpus
        if hasattr(reference_corpus, 'similarity'):
            similarity = reference_corpus.similarity(text)
            return 1.0 - similarity
        else:
            return self._heuristic_originality(text)
            
    def _heuristic_originality(self, text: str) -> float:
        """Calculate originality using heuristics."""
        score = 0.5  # Start with neutral
        
        # Bonus for creative language use
        creative_indicators = [
            "metaphor", "analogy", "imagine", "creative", "innovative",
            "unique", "original", "novel", "unprecedented"
        ]
        
        text_lower = text.lower()
        creative_count = sum(1 for word in creative_indicators 
                           if word in text_lower)
        score += min(0.3, creative_count * 0.1)
        
        # Bonus for unusual word combinations
        words = text.lower().split()
        if len(words) >= 2:
            # Simple measure: count of rare bigrams
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            # This is simplified - in practice, you'd use a corpus to determine rarity
            unusual_bigrams = len(set(bigrams))  # Placeholder
            score += min(0.2, unusual_bigrams * 0.01)
            
        return min(1.0, max(0.0, score))
        
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
