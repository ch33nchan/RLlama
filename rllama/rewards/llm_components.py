import numpy as np
from typing import Dict, List, Any, Optional, Union

class FactualityReward:
    def __init__(self, weight: float = 1.0, threshold: float = 0.7, hallucination_penalty: float = 2.0):
        self.name = "FactualityReward"
        self.weight = weight
        self.threshold = threshold
        self.hallucination_penalty = hallucination_penalty
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        factuality_score = state.get('factuality_score', 0.5)
        hallucination_score = state.get('hallucination_score', 0.0)
        
        # Penalize hallucinations
        hallucination_penalty = -self.hallucination_penalty * hallucination_score
        
        # Reward factuality above threshold, penalize below
        if factuality_score < self.threshold:
            factuality_component = -self.weight * (self.threshold - factuality_score) * 10
        else:
            factuality_component = self.weight * (factuality_score - self.threshold)
        
        return factuality_component + hallucination_penalty

class CoherenceReward:
    def __init__(self, weight: float = 1.0, min_score: float = 0.0, max_score: float = 1.0):
        self.name = "CoherenceReward"
        self.weight = weight
        self.min_score = min_score
        self.max_score = max_score
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        coherence_score = state.get('coherence_score', 0.5)
        normalized_score = (coherence_score - self.min_score) / (self.max_score - self.min_score)
        normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
        
        return self.weight * normalized_score

class RelevanceReward:
    def __init__(self, weight: float = 1.0, query_importance: float = 0.5):
        self.name = "RelevanceReward"
        self.weight = weight
        self.query_importance = query_importance
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        relevance_score = state.get('relevance_score', 0.5)
        query_match = state.get('query_match', 0.5)
        
        combined_score = (1 - self.query_importance) * relevance_score + self.query_importance * query_match
        return self.weight * combined_score

class HelpfulnessReward:
    def __init__(self, weight: float = 1.0, min_threshold: float = 0.3):
        self.name = "HelpfulnessReward"
        self.weight = weight
        self.min_threshold = min_threshold
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        helpfulness_score = state.get('helpfulness_score', 0.5)
        
        # Strong penalty for unhelpful responses
        if helpfulness_score < self.min_threshold:
            return -self.weight * (1.0 - helpfulness_score/self.min_threshold) * 5
        
        return self.weight * helpfulness_score

class HarmlessnessReward:
    def __init__(self, weight: float = 1.0, toxicity_penalty: float = 3.0):
        self.name = "HarmlessnessReward"
        self.weight = weight
        self.toxicity_penalty = toxicity_penalty
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        toxicity_score = state.get('toxicity_score', 0.0)
        harmlessness_score = state.get('harmlessness_score', 1.0 - toxicity_score)
        
        # Exponential penalty for toxicity
        if toxicity_score > 0.1:
            toxicity_penalty = -self.toxicity_penalty * (np.exp(toxicity_score * 2) - 1)
        else:
            toxicity_penalty = 0
        
        harmlessness_reward = self.weight * harmlessness_score
        
        return harmlessness_reward + toxicity_penalty

class ConcisionReward:
    def __init__(self, weight: float = 1.0, target_length: int = 200, tolerance: int = 100):
        self.name = "ConcisionReward"
        self.weight = weight
        self.target_length = target_length
        self.tolerance = tolerance
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        response_length = state.get('response_length', 0)
        
        # Calculate distance from target length
        distance = abs(response_length - self.target_length)
        
        # No penalty within tolerance
        if distance <= self.tolerance:
            return 0.0
        
        # Quadratic penalty for being too far from target
        penalty = -self.weight * ((distance - self.tolerance) / self.target_length) ** 2
        
        return penalty

class DiversityReward:
    def __init__(self, weight: float = 1.0, repetition_penalty: float = 1.5):
        self.name = "DiversityReward"
        self.weight = weight
        self.repetition_penalty = repetition_penalty
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        vocabulary_diversity = state.get('vocabulary_diversity', 0.5)
        repetition_score = state.get('repetition_score', 0.0)
        
        diversity_reward = self.weight * vocabulary_diversity
        repetition_penalty = -self.repetition_penalty * repetition_score
        
        return diversity_reward + repetition_penalty

class GroundingReward:
    def __init__(self, weight: float = 1.0, citation_bonus: float = 0.2, min_citations: int = 0):
        self.name = "GroundingReward"
        self.weight = weight
        self.citation_bonus = citation_bonus
        self.min_citations = min_citations
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        grounding_score = state.get('grounding_score', 0.5)
        citation_count = state.get('citation_count', 0)
        
        # Base reward for grounding
        base_reward = self.weight * grounding_score
        
        # Bonus for citations above minimum
        citation_bonus = self.citation_bonus * max(0, citation_count - self.min_citations)
        
        return base_reward + citation_bonus

class AlignmentReward:
    def __init__(self, weight: float = 1.0, 
                 factuality_importance: float = 0.3,
                 harmlessness_importance: float = 0.4,
                 helpfulness_importance: float = 0.3):
        self.name = "AlignmentReward"
        self.weight = weight
        self.factuality_importance = factuality_importance
        self.harmlessness_importance = harmlessness_importance
        self.helpfulness_importance = helpfulness_importance
    
    def calculate(self, state: Dict[str, Any], action: Any = None) -> float:
        factuality_score = state.get('factuality_score', 0.5)
        harmlessness_score = state.get('harmlessness_score', 1.0 - state.get('toxicity_score', 0.0))
        helpfulness_score = state.get('helpfulness_score', 0.5)
        
        # Combine the three pillars of alignment
        alignment_score = (
            self.factuality_importance * factuality_score +
            self.harmlessness_importance * harmlessness_score +
            self.helpfulness_importance * helpfulness_score
        )
        
        return self.weight * alignment_score