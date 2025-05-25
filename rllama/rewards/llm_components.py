import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple # Added Tuple here
from .base import BaseReward # Ensure BaseReward is imported

class FactualityReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, threshold: float = 0.7, hallucination_penalty: float = 2.0): # Add name, call super
        super().__init__(name, weight)
        self.threshold = threshold
        self.hallucination_penalty = hallucination_penalty
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        factuality_score = context.get('factuality_score', 0.5)
        hallucination_score = context.get('hallucination_score', 0.0)
        
        # Penalize hallucinations
        hallucination_penalty_val = -self.hallucination_penalty * hallucination_score
        
        # Reward factuality above threshold, penalize below
        if factuality_score < self.threshold:
            factuality_component = -self.weight * (self.threshold - factuality_score) * 10
        else:
            factuality_component = self.weight * (factuality_score - self.threshold)
        
        return factuality_component + hallucination_penalty_val

    def reset(self): # Add reset method
        pass

class CoherenceReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, min_score: float = 0.0, max_score: float = 1.0, **kwargs): # Add name, call super
        super().__init__(name, weight)
        self.min_score = min_score
        self.max_score = max_score
        # Ignore any extra kwargs to make the component more flexible
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        coherence_score = context.get('coherence_score', 0.5)
        # Ensure division by zero is handled if min_score can equal max_score
        if (self.max_score - self.min_score) == 0:
            normalized_score = 0.5 if coherence_score == self.min_score else (1.0 if coherence_score > self.min_score else 0.0)
        else:
            normalized_score = (coherence_score - self.min_score) / (self.max_score - self.min_score)
        
        normalized_score = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
        
        return self.weight * normalized_score

    def reset(self): # Add reset method
        pass

class RelevanceReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, query_importance: float = 0.5): # Add name, call super
        super().__init__(name, weight)
        self.query_importance = query_importance
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        relevance_score = context.get('relevance_score', 0.5)
        query_match = context.get('query_match', 0.5)
        
        combined_score = (1 - self.query_importance) * relevance_score + self.query_importance * query_match
        return self.weight * combined_score

    def reset(self): # Add reset method
        pass

class HelpfulnessReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, min_threshold: float = 0.3): # Add name, call super
        super().__init__(name, weight)
        self.min_threshold = min_threshold
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        helpfulness_score = context.get('helpfulness_score', 0.5)
        
        # Strong penalty for unhelpful responses
        if helpfulness_score < self.min_threshold:
            # Avoid division by zero if min_threshold can be 0
            penalty_factor = 0.0
            if self.min_threshold > 0: # Ensure min_threshold is not zero before division
                penalty_factor = (1.0 - helpfulness_score / self.min_threshold)
            return -self.weight * penalty_factor * 5 # Apply penalty factor
        
        return self.weight * helpfulness_score

    def reset(self): # Add reset method
        pass

class HarmlessnessReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, toxicity_penalty: float = 3.0): # Add name, call super
        super().__init__(name, weight)
        self.toxicity_penalty_factor = toxicity_penalty # Use a more descriptive name
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        toxicity_score = context.get('toxicity_score', 0.0)
        # Assuming harmlessness_score is inversely related or a separate metric
        # If it's just 1 - toxicity, then it's redundant with toxicity_score
        harmlessness_score = context.get('harmlessness_score', 1.0 - toxicity_score) 
        
        current_toxicity_penalty = 0.0 # Initialize penalty
        # Exponential penalty for toxicity
        if toxicity_score > 0.1: # Apply penalty only if toxicity exceeds a small threshold
            # The np.exp can grow very fast. Consider scaling or capping.
            # Example: np.exp(toxicity_score * 2) might be too aggressive.
            # Let's use a simpler scaling for now or ensure toxicity_score is well-bounded (e.g., 0-1)
            current_toxicity_penalty = -self.toxicity_penalty_factor * (np.exp(min(toxicity_score, 1.0) * 2) - 1) # Cap toxicity_score for exp
        
        # The reward should be based on harmlessness, then apply toxicity penalty
        # If harmlessness_score is just 1-toxicity, then:
        # reward = self.weight * (1 - toxicity_score) + current_toxicity_penalty
        # If harmlessness_score is an independent measure:
        reward = self.weight * harmlessness_score + current_toxicity_penalty
        
        return reward

    def reset(self): # Add reset method
        pass

class ConcisionReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, target_length: int = 200, tolerance: int = 100, **kwargs): # Add name, call super
        super().__init__(name, weight)
        self.target_length = target_length
        self.tolerance = tolerance
        # Ignore any extra kwargs
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        # Assuming 'response_text' is in context for length calculation
        response_text = context.get('response_text', '')
        # You might want to count words or characters based on your needs
        # response_length = len(response_text.split()) # Word count
        response_length = context.get('response_length', len(response_text)) # Or use pre-calculated length if available

        # Calculate distance from target length
        distance = abs(response_length - self.target_length)
        
        # No penalty within tolerance
        if distance <= self.tolerance:
            return 0.0 # Or a small positive reward for being within tolerance
        
        # Quadratic penalty for being too far from target
        # Ensure target_length is not zero to avoid division by zero
        if self.target_length == 0:
            penalty = -self.weight # Max penalty if target is 0 and not met
        else:
            penalty = -self.weight * ((distance - self.tolerance) / self.target_length) ** 2
        
        return penalty

    def reset(self): # Add reset method
        pass

class DiversityReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, repetition_penalty: float = 1.5, **kwargs): # Add name, call super
        super().__init__(name, weight)
        self.repetition_penalty_factor = repetition_penalty
        # Ignore any extra kwargs
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        vocabulary_diversity = context.get('vocabulary_diversity', 0.5) # e.g., unique words / total words
        repetition_score = context.get('repetition_score', 0.0) # e.g., based on n-gram repetition
        
        diversity_reward_val = self.weight * vocabulary_diversity
        repetition_penalty_val = -self.repetition_penalty_factor * repetition_score
        
        return diversity_reward_val + repetition_penalty_val

    def reset(self): # Add reset method
        pass

class GroundingReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, citation_bonus: float = 0.2, min_citations: int = 0): # Add name, call super
        super().__init__(name, weight)
        self.citation_bonus_factor = citation_bonus # Use a more descriptive name
        self.min_citations_threshold = min_citations # Use a more descriptive name
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        grounding_score = context.get('grounding_score', 0.5) # e.g., semantic similarity to source documents
        citation_count = context.get('citation_count', 0)
        
        # Base reward for grounding
        base_reward_val = self.weight * grounding_score
        
        # Bonus for citations above minimum
        citation_bonus_val = self.citation_bonus_factor * max(0, citation_count - self.min_citations_threshold)
        
        return base_reward_val + citation_bonus_val

    def reset(self): # Add reset method
        pass

class AlignmentReward(BaseReward): # Inherit from BaseReward
    def __init__(self, name: str, weight: float = 1.0, # Add name, call super
                 factuality_importance: float = 0.3,
                 harmlessness_importance: float = 0.4,
                 helpfulness_importance: float = 0.3):
        super().__init__(name, weight)
        self.factuality_importance = factuality_importance
        self.harmlessness_importance = harmlessness_importance
        self.helpfulness_importance = helpfulness_importance
    
    def calculate(self, context: Dict[str, Any]) -> float: # Changed 'state' to 'context', removed 'action'
        # These scores should ideally be in [0, 1] or a consistent range
        factuality_score = context.get('factuality_score', 0.5)
        # Harmlessness might be 1 - toxicity_score, or an independent measure
        harmlessness_score = context.get('harmlessness_score', 1.0 - context.get('toxicity_score', 0.0))
        helpfulness_score = context.get('helpfulness_score', 0.5)
        
        # Combine the three pillars of alignment
        # Ensure importance factors sum to 1 if they are meant to be weights in a convex combination
        # total_importance = self.factuality_importance + self.harmlessness_importance + self.helpfulness_importance
        # if total_importance == 0: alignment_score = 0.0
        # else:
        #     alignment_score = (
        #         (self.factuality_importance / total_importance) * factuality_score +
        #         (self.harmlessness_importance / total_importance) * harmlessness_score +
        #         (self.helpfulness_importance / total_importance) * helpfulness_score
        #     )
        # Simpler approach if importances are just scaling factors:
        alignment_score = (
            self.factuality_importance * factuality_score +
            self.harmlessness_importance * harmlessness_score +
            self.helpfulness_importance * helpfulness_score
        )
        
        return self.weight * alignment_score

    def reset(self): # Add reset method
        pass

import re # For heuristic-based checks
from .base import BaseReward # Already imported above

class SelfConsistencyReward(BaseReward): # Changed BaseRewardComponent to BaseReward
    """
    Rewards the LLM for maintaining self-consistency within its response
    and potentially across a dialogue history.
    It can use a user-provided LLM judge or fallback to heuristics.
    """
    def __init__(self, 
                 weight: float = 1.0, 
                 judge_llm_client: Optional[Any] = None, 
                 check_history: bool = False,
                 judge_prompt_template: Optional[str] = None,
                 heuristic_penalty: float = 1.0,
                 judge_positive_keywords: List[str] = ["yes", "consistent", "coherent"],
                 judge_negative_keywords: List[str] = ["no", "inconsistent", "contradictory", "contradicts"]):
        """
        Initializes the SelfConsistencyReward component.

        Args:
            weight (float): The weight of this reward component.
            judge_llm_client (Optional[Any]): An optional client object for an LLM judge.
                Expected to have a `generate(prompt: str, **kwargs) -> str` method.
            check_history (bool): If True, checks consistency against dialogue history.
            judge_prompt_template (Optional[str]): A custom prompt template for the LLM judge.
                Must contain a placeholder {text_to_evaluate}.
            heuristic_penalty (float): Penalty applied for each heuristic contradiction found.
            judge_positive_keywords (List[str]): Keywords in judge's response indicating consistency.
            judge_negative_keywords (List[str]): Keywords in judge's response indicating inconsistency.
        """
        super().__init__(weight)
        self.judge_llm_client = judge_llm_client
        self.check_history = check_history
        self.heuristic_penalty = heuristic_penalty
        self.judge_positive_keywords = [kw.lower() for kw in judge_positive_keywords]
        self.judge_negative_keywords = [kw.lower() for kw in judge_negative_keywords]

        if judge_prompt_template:
            if "{text_to_evaluate}" not in judge_prompt_template:
                raise ValueError("judge_prompt_template must contain '{text_to_evaluate}' placeholder.")
            self.judge_prompt_template = judge_prompt_template
        else:
            self.judge_prompt_template = (
                "Please evaluate the following text for internal self-consistency. "
                "Consider if there are any statements that contradict each other, "
                "either directly or implicitly. Respond with 'Consistent' or 'Inconsistent', "
                "followed by a brief explanation.\n\n"
                "Text to evaluate:\n{text_to_evaluate}\n\n"
                "Evaluation:"
            )

    def _get_text_to_evaluate(self, generated_text: str, context: Dict[str, Any]) -> str:
        text_to_evaluate = generated_text
        if self.check_history and 'dialogue_history' in context and context['dialogue_history']:
            history_parts = []
            for turn in context['dialogue_history']:
                # Assuming turn is a dict with 'role' and 'content'
                if isinstance(turn, dict) and 'content' in turn:
                    history_parts.append(f"{turn.get('role', 'user')}: {turn['content']}")
            if history_parts:
                history_str = "\n".join(history_parts)
                text_to_evaluate = f"Previous conversation:\n{history_str}\n\nCurrent response to evaluate:\n{generated_text}"
        return text_to_evaluate

    def _parse_judge_response(self, judge_response: str) -> float:
        """
        Parses the judge's textual response into a numerical score.
        Returns 1.0 for consistent, -1.0 for inconsistent, 0.0 for uncertain/unclear.
        """
        response_lower = judge_response.lower()
        
        has_positive = any(kw in response_lower for kw in self.judge_positive_keywords)
        has_negative = any(kw in response_lower for kw in self.judge_negative_keywords)

        if has_positive and not has_negative:
            return 1.0  # Consistent
        elif has_negative and not has_positive:
            return -1.0 # Inconsistent
        elif has_negative and has_positive: # Ambiguous response
            # Prioritize negative if both appear, could be "Yes, it is inconsistent"
            # Or, could be "It is not inconsistent" (double negative) - this simple check might fail.
            # More sophisticated parsing might be needed for complex judge outputs.
            # For now, if both appear, lean towards inconsistency or neutral.
            # Let's check for "not inconsistent" type patterns
            if any(f"not {neg_kw}" in response_lower for neg_kw in self.judge_negative_keywords):
                 # If "not inconsistent" is found, and no other negative kw, treat as positive
                 if not any(neg_kw in response_lower for neg_kw in self.judge_negative_keywords if f"not {neg_kw}" not in response_lower):
                    return 1.0 
            return -0.5 # Ambiguous, leaning negative
        
        # If no clear keywords, consider it uncertain
        return 0.0  # Uncertain or unable to parse

    def _heuristic_consistency_check(self, text: str) -> float:
        """
        Performs basic heuristic checks for self-contradictions.
        Returns a penalty score (0 or negative).
        """
        penalty = 0.0
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        # Example Heuristic 1: Direct negation (e.g., "X is Y" and "X is not Y")
        # This is a simplified example. Robust NLP would be needed for general cases.
        # Looking for patterns like "A is B" and "A is not B" or "A is B. A is not B."
        # This requires more sophisticated NLP to do robustly.
        # For a simple illustration:
        statements = {} # Store positive statements to check against negations

        for sentence in sentences:
            # Simple pattern: "entity is attribute" vs "entity is not attribute"
            match_positive = re.match(r"(\w+(?:\s\w+)*)\s+is\s+([\w\s]+)", sentence, re.IGNORECASE)
            match_negative = re.match(r"(\w+(?:\s\w+)*)\s+is\s+not\s+([\w\s]+)", sentence, re.IGNORECASE)

            if match_positive:
                entity, attribute = match_positive.groups()
                entity = entity.lower().strip()
                attribute = attribute.lower().strip()
                if statements.get(entity) == f"not {attribute}":
                    penalty -= self.heuristic_penalty
                statements[entity] = attribute
            
            elif match_negative:
                entity, attribute = match_negative.groups()
                entity = entity.lower().strip()
                attribute = attribute.lower().strip() # attribute here is what it's NOT
                if statements.get(entity) == attribute: # attribute here is the positive form
                    penalty -= self.heuristic_penalty
                statements[entity] = f"not {attribute}"
        
        # Example Heuristic 2: Contradictory claims about quantities (very simplified)
        # "There are 5 apples" and "There are 3 apples"
        quantity_claims = {}
        for sentence in sentences:
            match_quantity = re.search(r"there (?:are|is|were|was)\s+(\d+)\s+([\w\s]+)", sentence, re.IGNORECASE)
            if match_quantity:
                number, item = match_quantity.groups()
                item = item.lower().strip().rstrip('s') # normalize item
                if item in quantity_claims and quantity_claims[item] != number:
                    penalty -= self.heuristic_penalty
                quantity_claims[item] = number
                
        return penalty

    def calculate_reward(self, prompt: str, generated_text: str, context: Dict[str, Any]) -> float:
        """
        Calculates the reward based on the self-consistency of the generated_text.

        Args:
            prompt (str): The input prompt to the LLM. (Unused in this component directly)
            generated_text (str): The text generated by the LLM.
            context (dict): A dictionary containing additional context, which might include:
                            'dialogue_history': A list of previous turns.
                            'llm_judge_kwargs': Additional kwargs for the judge_llm_client.generate().

        Returns:
            float: The calculated reward.
        """
        if not generated_text.strip():
            return 0.0 # No text to evaluate

        text_to_evaluate = self._get_text_to_evaluate(generated_text, context)
        
        score = 0.0

        if self.judge_llm_client:
            try:
                judge_prompt = self.judge_prompt_template.format(text_to_evaluate=text_to_evaluate)
                llm_judge_kwargs = context.get('llm_judge_kwargs', {})
                judge_response = self.judge_llm_client.generate(judge_prompt, **llm_judge_kwargs)
                score = self._parse_judge_response(judge_response)
            except Exception as e:
                # Log error, and potentially fall back to heuristics or return neutral score
                print(f"Error using LLM judge for SelfConsistencyReward: {e}")
                # Fallback to heuristics if judge fails
                score = self._heuristic_consistency_check(text_to_evaluate)
        else:
            # Use heuristic-based checks if no judge LLM
            score = self._heuristic_consistency_check(text_to_evaluate)
            # Heuristics return penalties (0 or negative). We might want to scale this
            # or provide a small positive base if no contradictions are found.
            # For now, if score is 0 (no penalty), it means no contradictions found by heuristics.
            # Let's give a small positive reward if no heuristic contradictions are found.
            if score == 0.0:
                score = 0.1 # Small positive reward for passing heuristic checks
        
        # The calculate method is expected by the RewardComposer
        # The parameters for calculate should match BaseReward's calculate method
        # which is `calculate(self, context: Dict[str, Any]) -> float:`
        # We need to adapt this. For now, let's assume generated_text is in context.
        # This component's `calculate_reward` method is specific.
        # We need to align it with the `BaseReward` interface or call it internally.

        # For now, let's assume `generated_text` and `prompt` are passed via `context`
        # to align with the `BaseReward.calculate` signature.
        # This might require changes in how TRLRllamaRewardProcessor populates the context.
        # Alternatively, SelfConsistencyReward's `calculate` method needs to be the entry point.

        # Let's rename `calculate_reward` to `calculate` and adjust its signature
        # to match BaseReward.
        # The actual logic will be called from within this `calculate` method.
        # This is a significant refactor of this specific component.
        # For now, I will keep calculate_reward and assume it's called internally
        # or the BaseReward interface is more flexible.
        # Given the error is NameError, the immediate fix is the inheritance.
        # The method signature mismatch is a subsequent issue.

        return score * self.weight # This line should be part of calculate_reward

    # This method should be the one defined in BaseReward
    def calculate(self, context: Dict[str, Any]) -> float:
        prompt = context.get("prompt", "")
        generated_text = context.get("response_text", "") # Assuming response_text is the generated_text
        # Call the original logic
        return self.calculate_reward(prompt, generated_text, context)

    def reset(self): # Add reset method to conform to BaseReward
        pass


class ChainOfThoughtValidityReward(BaseReward):
    def __init__(self,
                 weight: float = 1.0,
                 judge_llm_client: Optional[Any] = None,
                 judge_prompt_template: Optional[str] = None,
                 cot_extraction_regex: Optional[str] = None,
                 min_cot_steps_heuristic: int = 2,
                 heuristic_logical_gap_penalty: float = 0.5,
                 heuristic_irrelevance_penalty: float = 0.5,
                 heuristic_missing_cot_penalty: float = 1.0,
                 judge_positive_keywords: List[str] = ["valid", "sound", "logical", "correct steps"],
                 judge_negative_keywords: List[str] = ["invalid", "unsound", "illogical", "flawed", "missing steps", "incorrect reasoning"]):
        super().__init__(weight)
        self.judge_llm_client = judge_llm_client
        self.cot_extraction_regex = cot_extraction_regex
        self.min_cot_steps_heuristic = min_cot_steps_heuristic
        self.heuristic_logical_gap_penalty = heuristic_logical_gap_penalty
        self.heuristic_irrelevance_penalty = heuristic_irrelevance_penalty
        self.heuristic_missing_cot_penalty = heuristic_missing_cot_penalty
        self.judge_positive_keywords = [kw.lower() for kw in judge_positive_keywords]
        self.judge_negative_keywords = [kw.lower() for kw in judge_negative_keywords]

        if judge_prompt_template:
            if "{original_prompt}" not in judge_prompt_template or \
               "{chain_of_thought_and_answer}" not in judge_prompt_template:
                raise ValueError("judge_prompt_template must contain '{original_prompt}' and '{chain_of_thought_and_answer}' placeholders.")
            self.judge_prompt_template = judge_prompt_template
        else:
            self.judge_prompt_template = (
                "Original Prompt:\n{original_prompt}\n\n"
                "Generated Response (including Chain of Thought and Final Answer):\n{chain_of_thought_and_answer}\n\n"
                "Please evaluate the validity and soundness of the chain of thought presented in the response. "
                "Consider if the reasoning steps are logical, relevant to the prompt, and correctly lead to the final answer. "
                "Respond with 'Valid CoT' or 'Invalid CoT', followed by a brief explanation of any flaws or strengths."
                "\n\nEvaluation:"
            )

    def _extract_cot_and_answer(self, generated_text: str) -> Tuple[Optional[str], str]:
        if self.cot_extraction_regex:
            match = re.search(self.cot_extraction_regex, generated_text, re.DOTALL | re.IGNORECASE)
            if match:
                cot = match.group(1).strip()
                
                answer_part_after_cot = generated_text[match.end():].strip()
                if not answer_part_after_cot: # If regex captures up to the end, CoT might include answer
                    return cot, cot 
                return cot, answer_part_after_cot 
        
        common_cot_markers = [
            "Let's think step by step:",
            "Here's my thinking process:",
            "Step 1:",
            "Reasoning:"
        ]
        text_lower = generated_text.lower()
        for marker in common_cot_markers:
            if marker.lower() in text_lower:
                start_index = text_lower.find(marker.lower())
                
                # Try to find a "Final Answer:" or similar marker for the end of CoT
                final_answer_markers = ["Final Answer:", "The answer is:", "Therefore, the final answer is:"]
                end_cot_index = -1
                extracted_answer = ""

                for fa_marker in final_answer_markers:
                    fa_marker_lower = fa_marker.lower()
                    if fa_marker_lower in text_lower[start_index:]:
                        end_cot_index = text_lower.find(fa_marker_lower, start_index)
                        extracted_answer = generated_text[end_cot_index + len(fa_marker):].strip()
                        break
                
                if end_cot_index != -1:
                    cot = generated_text[start_index : end_cot_index].strip()
                    return cot, extracted_answer if extracted_answer else generated_text[end_cot_index:].strip()
                else: # No clear final answer marker after CoT marker
                    return generated_text[start_index:].strip(), generated_text # Assume CoT is till end, answer is part of it or whole text
        
        return None, generated_text


    def _parse_judge_response(self, judge_response: str) -> float:
        response_lower = judge_response.lower()
        has_positive = any(kw in response_lower for kw in self.judge_positive_keywords)
        has_negative = any(kw in response_lower for kw in self.judge_negative_keywords)

        if has_positive and not has_negative:
            return 1.0
        elif has_negative and not has_positive:
            return -1.0
        elif has_negative and has_positive:
            if any(f"not {neg_kw}" in response_lower for neg_kw in self.judge_negative_keywords):
                 if not any(neg_kw in response_lower for neg_kw in self.judge_negative_keywords if f"not {neg_kw}" not in response_lower):
                    return 1.0
            return -0.5
        return 0.0

    def _heuristic_cot_check(self, original_prompt: str, chain_of_thought: Optional[str], final_answer: str) -> float:
        penalty = 0.0

        if not chain_of_thought:
            return -self.heuristic_missing_cot_penalty

        cot_steps = [s.strip() for s in chain_of_thought.split('\n') if s.strip() and len(s.split()) > 2]
        if not cot_steps: # Handles case where CoT is present but effectively empty after splitting
             cot_steps = [s.strip() for s in re.split(r'\.\s+|\;\s+|implies\s+|therefore\s+', chain_of_thought) if s.strip() and len(s.split()) > 1]


        if len(cot_steps) < self.min_cot_steps_heuristic:
            penalty -= self.heuristic_missing_cot_penalty / 2 

        prompt_keywords = set(re.findall(r'\b\w+\b', original_prompt.lower()))

        if not cot_steps: # If still no steps after alternative splitting
            return penalty - self.heuristic_irrelevance_penalty # Penalize if CoT is empty or unparseable

        for i, step in enumerate(cot_steps):
            step_lower = step.lower()
            step_keywords = set(re.findall(r'\b\w+\b', step_lower))

            if not prompt_keywords.intersection(step_keywords):
                if i == 0: 
                    penalty -= self.heuristic_irrelevance_penalty
            
            if i > 0:
                prev_step_keywords = set(re.findall(r'\b\w+\b', cot_steps[i-1].lower()))
                if not step_keywords.intersection(prev_step_keywords) and \
                   not step_keywords.intersection(prompt_keywords):
                    penalty -= self.heuristic_logical_gap_penalty
        
        reasoning_keywords = ["because", "therefore", "hence", "thus", "since", "so", "consequently", "as a result", "implies", "leads to"]
        has_reasoning_indicator = any(rk in chain_of_thought.lower() for rk in reasoning_keywords)
        if not has_reasoning_indicator and len(cot_steps) > 1:
            penalty -= self.heuristic_logical_gap_penalty / 2 

        if final_answer.strip() and chain_of_thought.strip():
            answer_keywords = set(re.findall(r'\b\w+\b', final_answer.lower()))
            last_cot_step_keywords = set(re.findall(r'\b\w+\b', cot_steps[-1].lower()))
            if not answer_keywords.intersection(last_cot_step_keywords) and \
               not answer_keywords.intersection(prompt_keywords):
                 penalty -= self.heuristic_logical_gap_penalty
        
        return max(penalty, -2.0) # Cap max penalty from heuristics

    def calculate_reward(self, prompt: str, generated_text: str, context: Dict[str, Any]) -> float:
        if not generated_text.strip():
            return 0.0

        chain_of_thought, final_answer_or_full_text = self._extract_cot_and_answer(generated_text)
        
        text_for_judge = generated_text 
        if chain_of_thought and final_answer_or_full_text != chain_of_thought :
             text_for_judge = f"Chain of Thought:\n{chain_of_thought}\n\nFinal Answer:\n{final_answer_or_full_text}"
        elif chain_of_thought: # CoT might contain the answer or is the whole relevant part
             text_for_judge = chain_of_thought


        score = 0.0
        if self.judge_llm_client:
            try:
                judge_api_prompt = self.judge_prompt_template.format(
                    original_prompt=prompt,
                    chain_of_thought_and_answer=text_for_judge
                )
                llm_judge_kwargs = context.get('llm_judge_kwargs', {})
                judge_response = self.judge_llm_client.generate(judge_api_prompt, **llm_judge_kwargs)
                score = self._parse_judge_response(judge_response)
            except Exception as e:
                print(f"Error using LLM judge for ChainOfThoughtValidityReward: {e}")
                score = self._heuristic_cot_check(prompt, chain_of_thought, final_answer_or_full_text)
        else:
            score = self._heuristic_cot_check(prompt, chain_of_thought, final_answer_or_full_text)
            if score == 0.0 and chain_of_thought: 
                score = 0.1 
            elif score == 0.0 and not chain_of_thought: # No CoT found, no penalty from heuristic (means it wasn't expected)
                score = 0.0 # Neutral if CoT not applicable/found and not penalized

        return score * self.weight

class InstructionFollowingComplexityReward(BaseReward):
    def __init__(self,
                 weight: float = 1.0,
                 judge_llm_client: Optional[Any] = None,
                 judge_prompt_template: Optional[str] = None,
                 heuristic_violation_penalty: float = 0.75,
                 heuristic_partial_adherence_bonus: float = 0.2,
                 judge_positive_keywords: List[str] = ["fully adhered", "followed all instructions", "correctly executed"],
                 judge_negative_keywords: List[str] = ["failed to follow", "missed instruction", "violated constraint", "incorrect format"]):
        super().__init__(weight)
        self.judge_llm_client = judge_llm_client
        self.heuristic_violation_penalty = heuristic_violation_penalty
        self.heuristic_partial_adherence_bonus = heuristic_partial_adherence_bonus
        self.judge_positive_keywords = [kw.lower() for kw in judge_positive_keywords]
        self.judge_negative_keywords = [kw.lower() for kw in judge_negative_keywords]

        if judge_prompt_template:
            if "{original_instruction}" not in judge_prompt_template or \
               "{generated_response}" not in judge_prompt_template:
                raise ValueError("judge_prompt_template must contain '{original_instruction}' and '{generated_response}' placeholders.")
            self.judge_prompt_template = judge_prompt_template
        else:
            self.judge_prompt_template = (
                "Original Instruction:\n{original_instruction}\n\n"
                "Generated Response:\n{generated_response}\n\n"
                "Please evaluate how well the generated response adheres to all aspects of the original instruction, "
                "including any explicit or implicit constraints, formatting requirements, and negative constraints (e.g., 'do not mention X'). "
                "Respond with 'Fully Adhered' or 'Partially Adhered' or 'Failed to Adhere', followed by a brief explanation."
                "\n\nEvaluation:"
            )

    def _parse_judge_response(self, judge_response: str) -> float:
        response_lower = judge_response.lower()
        
        is_positive = any(kw in response_lower for kw in self.judge_positive_keywords)
        is_negative = any(kw in response_lower for kw in self.judge_negative_keywords)

        if "fully adhered" in response_lower or ("adhered" in response_lower and not is_negative): # Prioritize "fully adhered"
            return 1.0
        elif "partially adhered" in response_lower and not is_negative:
            return 0.5
        elif is_negative: # "failed to adhere" or other negative keywords
            return -1.0
        
        if is_positive and not is_negative : # General positive keyword
            return 0.7 # Slightly less than fully adhered if not explicit
        
        return 0.0


    def _heuristic_instruction_check(self, original_instruction: str, generated_response: str) -> float:
        score = 0.0
        instruction_lower = original_instruction.lower()
        response_lower = generated_response.lower()
        num_constraints_found = 0
        num_constraints_met = 0

        negative_constraints = re.findall(r"(?:do not|don't|avoid|must not) include ([\w\s]+?)(?:\.|;|$|,|and)", instruction_lower)
        negative_constraints += re.findall(r"(?:do not|don't|avoid|must not) mention ([\w\s]+?)(?:\.|;|$|,|and)", instruction_lower)
        for neg_constraint_phrase in negative_constraints:
            num_constraints_found += 1
            neg_constraint = neg_constraint_phrase.strip()
            if neg_constraint and neg_constraint in response_lower:
                score -= self.heuristic_violation_penalty
            else:
                num_constraints_met +=1
        
        positive_constraints = re.findall(r"(?:must include|ensure you provide|include|provide|mention) ([\w\s]+?)(?:\.|;|$|,|and)", instruction_lower)
        for pos_constraint_phrase in positive_constraints:
            num_constraints_found += 1
            pos_constraint = pos_constraint_phrase.strip()
            if pos_constraint and pos_constraint not in response_lower:
                score -= self.heuristic_violation_penalty
            elif pos_constraint:
                num_constraints_met +=1

        format_match = re.search(r"format as (json|xml|a list|bullet points|markdown)", instruction_lower)
        if format_match:
            num_constraints_found += 1
            format_type = format_match.group(1)
            met_format = False
            if format_type == "json":
                if (response_lower.startswith("{") and response_lower.endswith("}")) or \
                   (response_lower.startswith("[") and response_lower.endswith("]")):
                    met_format = True
            elif format_type == "xml":
                if response_lower.startswith("<") and response_lower.endswith(">") and ">" in response_lower[1:-1]:
                    met_format = True
            elif format_type == "a list" or format_type == "bullet points":
                if re.search(r"^\s*[\*\-\•]\s+", generated_response, re.MULTILINE) or \
                   re.search(r"^\s*\d+\.\s+", generated_response, re.MULTILINE):
                    met_format = True
            elif format_type == "markdown":
                 if re.search(r"#{1,6}\s|\*+[\w\s]+\*\*|__[\w\s]+__|`[\w\s]+`|\[.*\]\(.*\)|^\s*[\*\-\+]\s+", generated_response):
                    met_format = True
            
            if met_format:
                num_constraints_met +=1
            else:
                score -= self.heuristic_violation_penalty
        
        length_constraints = re.findall(r"(?:in|within|exactly|about|around)\s+(\d+)\s+(sentences?|words?|paragraphs?|points?)", instruction_lower)
        for num_str, unit in length_constraints:
            num_constraints_found += 1
            target_len = int(num_str)
            actual_len = 0
            if unit.startswith("sentence"):
                actual_len = len(re.findall(r"[^.!?]+[.!?]", generated_response))
            elif unit.startswith("word"):
                actual_len = len(generated_response.split())
            elif unit.startswith("paragraph"):
                actual_len = len(re.split(r"\n\s*\n", generated_response.strip()))
            elif unit.startswith("point"): # e.g. bullet points
                actual_len = len(re.findall(r"^\s*[\*\-\•\d\.]\s+", generated_response, re.MULTILINE))

            if actual_len > 0: # Only penalize if we could measure
                # Allow some tolerance, e.g., 20%
                tolerance = 0.2 * target_len
                if abs(actual_len - target_len) > max(2, tolerance): # max(2, tolerance) for small target_len
                    score -= self.heuristic_violation_penalty / 2 # Lesser penalty for length
                else:
                    num_constraints_met +=1
            else: # Could not determine length for the unit, or response was empty
                score -= self.heuristic_violation_penalty / 4 # Small penalty if unit was specified but not measurable

        if num_constraints_found == 0: # No specific constraints parsed by heuristics
            return 0.1 # Small default positive if no constraints to check

        if num_constraints_met == num_constraints_found and num_constraints_found > 0:
            score = self.heuristic_partial_adherence_bonus * 2 # Max bonus for full adherence
        elif num_constraints_met > 0:
            score += self.heuristic_partial_adherence_bonus * (num_constraints_met / num_constraints_found)
        
        return max(-1.0, min(1.0, score))


    def calculate_reward(self, prompt: str, generated_text: str, context: Dict[str, Any]) -> float:
        if not generated_text.strip():
            return -1.0 * self.weight # Penalize empty response if instruction was given

        original_instruction = prompt 
        
        score = 0.0
        if self.judge_llm_client:
            try:
                judge_api_prompt = self.judge_prompt_template.format(
                    original_instruction=original_instruction,
                    generated_response=generated_text
                )
                llm_judge_kwargs = context.get('llm_judge_kwargs', {})
                judge_response = self.judge_llm_client.generate(judge_api_prompt, **llm_judge_kwargs)
                score = self._parse_judge_response(judge_response)
            except Exception as e:
                print(f"Error using LLM judge for InstructionFollowingComplexityReward: {e}")
                score = self._heuristic_instruction_check(original_instruction, generated_text)
        else:
            score = self._heuristic_instruction_check(original_instruction, generated_text)

        return score * self.weight

# Add to __all__ if you have one:
# __all__ = [
#     "FactualityReward", "CoherenceReward", "RelevanceReward", "HelpfulnessReward",
#     "HarmlessnessReward", "ConcisionReward", "DiversityReward", "GroundingReward",
#     "AlignmentReward", "SelfConsistencyReward", "ChainOfThoughtValidityReward",
#     "InstructionFollowingComplexityReward", "ArgumentationQualityReward"
# ]