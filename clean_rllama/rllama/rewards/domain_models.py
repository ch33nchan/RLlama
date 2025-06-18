# rllama/rewards/domain_models.py

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import os
import json
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DomainRewardModel(ABC):
    """
    Base class for domain-specific reward models that can provide tailored rewards
    for specific industries or domains.
    """
    
    def __init__(self, 
                 name: str,
                 domain: str,
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize a domain reward model.
        
        Args:
            name: Name of the reward model.
            domain: Domain or industry this model is specialized for.
            model_path: Path to pre-trained model or model ID.
            device: Device to run inference on.
        """
        self.name = name
        self.domain = domain
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def initialize(self):
        """
        Lazy initialization of models to save resources.
        Only load models when needed.
        """
        if not self._initialized and self.model_path:
            self._load_model()
            self._initialized = True
    
    @abstractmethod
    def _load_model(self):
        """
        Load the actual model. Implementation depends on model type.
        """
        pass
    
    @abstractmethod
    def compute_reward(self, context: Dict[str, Any]) -> float:
        """
        Compute domain-specific reward based on context.
        
        Args:
            context: Context information for reward computation.
            
        Returns:
            Computed reward value.
        """
        pass
    
    def __call__(self, context: Dict[str, Any]) -> float:
        """Make the reward model callable for easier use"""
        if not self._initialized:
            self.initialize()
        return self.compute_reward(context)


class LLMJudgeRewardModel(DomainRewardModel):
    """
    Uses an LLM as a judge to provide domain-specific rewards.
    """
    
    def __init__(self, 
                 name: str,
                 domain: str,
                 model_path: str,
                 prompt_template: str,
                 criteria: List[str],
                 output_parser: Optional[Callable] = None,
                 device: str = "cpu"):
        """
        Initialize an LLM-based judgment reward model.
        
        Args:
            name: Name of the model.
            domain: Domain this model specializes in.
            model_path: Path to the LLM model to use.
            prompt_template: Template for prompting the model.
            criteria: List of criteria for judgment.
            output_parser: Function to parse model output into a reward value.
            device: Device to run the model on.
        """
        super().__init__(name, domain, model_path, device)
        self.prompt_template = prompt_template
        self.criteria = criteria
        self.output_parser = output_parser or self._default_parser
        
    def _load_model(self):
        """Load the LLM model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_reward(self, context: Dict[str, Any]) -> float:
        """Compute reward using LLM as judge"""
        # Ensure the model is loaded
        if not self._initialized:
            self.initialize()
        
        # Extract relevant information
        query = context.get("query", "")
        response = context.get("response", "")
        
        # Construct prompt for the LLM judge
        prompt = self.prompt_template.format(
            domain=self.domain,
            query=query,
            response=response,
            criteria="\n".join(f"- {criterion}" for criterion in self.criteria)
        )
        
        # Generate judgment
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=False
            )
        
        # Decode output
        judgment_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the part after the prompt
        if judgment_text.startswith(prompt):
            judgment_text = judgment_text[len(prompt):]
        
        # Parse output to get reward
        reward = self.output_parser(judgment_text)
        return reward
    
    def _default_parser(self, text: str) -> float:
        """Default parser that looks for ratings in the text"""
        # Look for patterns like "Rating: 8/10" or "Score: 4.5"
        import re
        
        # Try to find a rating of form X/10
        match = re.search(r'(\d+(?:\.\d+)?)\s*\/\s*10', text)
        if match:
            try:
                score = float(match.group(1))
                return score / 10.0  # Normalize to [0, 1]
            except ValueError:
                pass
        
        # Try to find a decimal rating
        match = re.search(r'rating[:|=]?\s*(\d+(?:\.\d+)?)', text.lower())
        if match:
            try:
                score = float(match.group(1))
                if score > 10:  # Assuming score is out of 100
                    return score / 100.0
                else:  # Assuming score is out of 10
                    return score / 10.0
            except ValueError:
                pass
        
        # If no numeric rating found, try sentiment analysis
        positive_terms = ["excellent", "good", "positive", "effective", "successful", "recommend"]
        negative_terms = ["poor", "bad", "negative", "ineffective", "unsuccessful", "avoid"]
        
        text_lower = text.lower()
        positive_count = sum(text_lower.count(term) for term in positive_terms)
        negative_count = sum(text_lower.count(term) for term in negative_terms)
        
        if positive_count > negative_count:
            return 0.7  # Positive sentiment
        elif negative_count > positive_count:
            return 0.3  # Negative sentiment
        else:
            return 0.5  # Neutral


class ReasoningRewardModel(DomainRewardModel):
    """
    Specialized reward model that evaluates the quality of reasoning in responses.
    Uses Chain-of-Thought evaluation.
    """
    
    def __init__(self,
                name: str,
                domain: str = "general",
                model_path: str = None,
                reasoning_criteria: Dict[str, float] = None,
                device: str = "cpu"):
        """
        Initialize a reasoning reward model.
        
        Args:
            name: Name of the model.
            domain: Domain for reasoning (e.g., "medical", "legal").
            model_path: Path to model or model ID.
            reasoning_criteria: Criteria for evaluating reasoning quality.
            device: Device to run model on.
        """
        super().__init__(name, domain, model_path, device)
        
        # Default criteria if none provided
        self.reasoning_criteria = reasoning_criteria or {
            "logical_coherence": 0.3,
            "evidence_support": 0.3,
            "step_by_step": 0.2,
            "alternative_views": 0.1,
            "conclusion_validity": 0.1
        }
    
    def _load_model(self):
        """Load models for reasoning evaluation"""
        # For reasoning evaluation, we can use a combination of models:
        # 1. A classifier for different reasoning components
        # 2. A model to detect logical fallacies
        
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Classifier for reasoning components
        if self.model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            # If no specific model provided, use a default one
            default_model = "facebook/bart-large-mnli"  # Natural language inference model
            self.model = AutoModelForSequenceClassification.from_pretrained(default_model).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(default_model)
    
    def compute_reward(self, context: Dict[str, Any]) -> float:
        """Compute reward based on reasoning quality"""
        if not self._initialized:
            self.initialize()
        
        # Extract response and query
        response = context.get("response", "")
        query = context.get("query", "")
        
        # Look for Chain-of-Thought patterns in the response
        cot_indicators = self._evaluate_chain_of_thought(response)
        
        # Evaluate coherence between query and response
        coherence_score = self._evaluate_coherence(query, response)
        
        # Evaluate evidence support
        evidence_score = self._evaluate_evidence(response)
        
        # Combine scores based on criteria weights
        total_score = (
            self.reasoning_criteria["logical_coherence"] * coherence_score +
            self.reasoning_criteria["evidence_support"] * evidence_score +
            self.reasoning_criteria["step_by_step"] * cot_indicators["steps_present"] +
            self.reasoning_criteria["alternative_views"] * cot_indicators["alternatives_considered"] +
            self.reasoning_criteria["conclusion_validity"] * cot_indicators["conclusion_supported"]
        )
        
        return total_score
    
    def _evaluate_chain_of_thought(self, text: str) -> Dict[str, float]:
        """
        Evaluate if the text contains chain-of-thought reasoning patterns
        """
        import re
        
        # Look for numbered steps or bullet points
        step_pattern = r'(?:\d+\.|\*|\-)\s+\w+'
        steps = re.findall(step_pattern, text)
        
        # Look for reasoning connectives
        connectives = ["therefore", "thus", "because", "so", "since", "as a result"]
        connective_count = sum(text.lower().count(conn) for conn in connectives)
        
        # Look for consideration of alternatives
        alternatives = ["alternatively", "on the other hand", "however", "but", "another possibility"]
        alternative_count = sum(text.lower().count(alt) for alt in alternatives)
        
        # Look for conclusion indicators
        conclusions = ["in conclusion", "to sum up", "finally", "consequently", "as we can see"]
        conclusion_indicators = sum(text.lower().count(conc) for conc in conclusions)
        
        # Compute scores
        steps_present = min(1.0, len(steps) / 3)  # Normalize, cap at 1.0
        has_connectives = min(1.0, connective_count / 2)
        alternatives_considered = min(1.0, alternative_count / 2)
        conclusion_supported = min(1.0, conclusion_indicators)
        
        return {
            "steps_present": steps_present,
            "has_connectives": has_connectives, 
            "alternatives_considered": alternatives_considered,
            "conclusion_supported": conclusion_supported
        }
    
    def _evaluate_coherence(self, premise: str, hypothesis: str) -> float:
        """
        Evaluate logical coherence between premise and hypothesis
        """
        if not premise or not hypothesis:
            return 0.5  # Neutral if either is missing
        
        # Use the NLI model for coherence
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # For NLI models: [contradiction, neutral, entailment]
            # We want high scores for entailment, medium for neutral, low for contradiction
            entailment_score = predictions[0, 2].item()
            contradiction_score = predictions[0, 0].item()
            
            # Higher score for entailment, lower for contradiction
            coherence_score = entailment_score + 0.5 * (1 - contradiction_score - entailment_score)
            
            return min(1.0, coherence_score)
    
    def _evaluate_evidence(self, text: str) -> float:
        """
        Evaluate if the text provides evidence for its claims
        """
        import re
        
        # Look for evidence indicators
        evidence_patterns = [
            r'research (shows|indicates|suggests)',
            r'according to [^\.]+',
            r'(studies|evidence) (show|suggest|indicate)',
            r'data (shows|indicates|suggests)',
            r'statistics (show|indicate|suggest)',
            r'example[s]? (of|include|are)',
        ]
        
        evidence_count = 0
        for pattern in evidence_patterns:
            evidence_count += len(re.findall(pattern, text.lower()))
        
        # Look for quantitative evidence
        number_pattern = r'\d+(\.\d+)?%?'
        numbers = re.findall(number_pattern, text)
        
        # Combine scores
        evidence_score = min(1.0, (evidence_count + len(numbers)) / 5)  # Cap at 1.0
        
        return evidence_score