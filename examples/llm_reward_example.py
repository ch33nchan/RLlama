import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllama.rewards.llm_components import (
    FactualityReward, CoherenceReward, RelevanceReward, 
    HelpfulnessReward, HarmlessnessReward, ConcisionReward,
    DiversityReward, GroundingReward, AlignmentReward
)
from rllama.rewards.composition import RewardComposer
from rllama.rewards.normalization import RewardNormalizer, AdaptiveNormalizer

def simulate_llm_response(prompt_type: str) -> Dict[str, Any]:
    """Simulate different types of LLM responses for testing reward components."""
    base_state = {
        'factuality_score': 0.8,
        'coherence_score': 0.7,
        'relevance_score': 0.75,
        'query_match': 0.6,
        'helpfulness_score': 0.65,
        'toxicity_score': 0.05,
        'response_length': 200,
        'repetition_score': 0.1,
        'vocabulary_diversity': 0.7,
        'grounding_score': 0.6,
        'citation_count': 2,
        'hallucination_score': 0.1
    }
    
    # Add noise to make it more realistic
    noise_factor = 0.1
    for key in base_state:
        if isinstance(base_state[key], float):
            base_state[key] = min(1.0, max(0.0, 
                                          base_state[key] + noise_factor * (np.random.random() - 0.5)))
    
    # Modify based on prompt type
    if prompt_type == "factual_error":
        base_state['factuality_score'] = 0.3
        base_state['hallucination_score'] = 0.6
    elif prompt_type == "toxic":
        base_state['toxicity_score'] = 0.7
        base_state['harmlessness_score'] = 0.2
    elif prompt_type == "irrelevant":
        base_state['relevance_score'] = 0.2
        base_state['query_match'] = 0.1
    elif prompt_type == "repetitive":
        base_state['repetition_score'] = 0.8
        base_state['vocabulary_diversity'] = 0.3
    elif prompt_type == "too_long":
        base_state['response_length'] = 2000
    elif prompt_type == "too_short":
        base_state['response_length'] = 20
        base_state['helpfulness_score'] = 0.3
    elif prompt_type == "perfect":
        for key in base_state:
            if key in ['toxicity_score', 'repetition_score', 'hallucination_score']:
                base_state[key] = 0.0
            elif isinstance(base_state[key], float):
                base_state[key] = 0.9
        base_state['response_length'] = 250
        base_state['citation_count'] = 5
    
    return base_state

def evaluate_response_types(reward_components: Dict[str, Any], normalizer: RewardNormalizer = None):
    response_types = [
        "standard", "factual_error", "toxic", "irrelevant", 
        "repetitive", "too_long", "too_short", "perfect"
    ]
    
    composer = RewardComposer(reward_components)
    
    results = {}
    component_results = {}
    
    for response_type in response_types:
        state = simulate_llm_response(response_type)
        
        # Calculate individual component rewards
        component_scores = {}
        for name, component in reward_components.items():
            score = component.calculate(state)
            if normalizer:
                score = normalizer.normalize(score, name)
            component_scores[name] = score
        
        # Calculate total reward
        total_reward = composer.calculate(state, None) # Context is None for this example
        if normalizer:
            total_reward = normalizer.normalize(total_reward, "total")
        
        results[response_type] = total_reward
        component_results[response_type] = component_scores
    
    return results, component_results

def visualize_rewards(results: Dict[str, float], component_results: Dict[str, Dict[str, float]], 
                     save_path: str = "llm_rewards.png"):
    response_types = list(results.keys())
    total_rewards = [results[rt] for rt in response_types]
    
    component_names = list(next(iter(component_results.values())).keys())
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot total rewards
    axes[0].bar(response_types, total_rewards, color='blue', alpha=0.7)
    axes[0].set_title('Total Reward by Response Type')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_xticklabels(response_types, rotation=45, ha="right")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot component rewards
    bar_width = 0.8 / len(component_names)
    x_indices = np.arange(len(response_types))
    
    for i, component in enumerate(component_names):
        component_values = [component_results[rt][component] for rt in response_types]
        # Offset x_positions for grouped bar chart
        x_positions = x_indices + (i - len(component_names)/2 + 0.5) * bar_width
        axes[1].bar(x_positions, component_values, width=bar_width, 
                   label=component, alpha=0.7)
    
    axes[1].set_title('Component Rewards by Response Type')
    axes[1].set_ylabel('Component Reward')
    axes[1].set_xticks(x_indices) # Set x-ticks to be at the center of groups
    axes[1].set_xticklabels(response_types, rotation=45, ha="right")
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3) # Adjusted legend position
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for legend
    plt.savefig(save_path)
    plt.close(fig)
    print(f"LLM reward visualization saved to {save_path}")
    
    return save_path

def compare_normalization_methods(reward_components: Dict[str, Any], save_path: str = "normalization_comparison.png"):
    response_types = [
        "standard", "factual_error", "toxic", "irrelevant", 
        "repetitive", "too_long", "too_short", "perfect"
    ]
    
    normalizers = {
        "None": None,
        "Standard": RewardNormalizer(method="standard"),
        "MinMax": RewardNormalizer(method="minmax"),
        "Robust": RewardNormalizer(method="robust"),
        "Adaptive": AdaptiveNormalizer() # AdaptiveNormalizer might need more steps to adapt
    }
    
    all_results = {}
    
    for norm_name, normalizer in normalizers.items():
        # For adaptive normalizer, simulate some steps to let it adapt
        if isinstance(normalizer, AdaptiveNormalizer):
            for _ in range(100): # Simulate 100 steps
                state = simulate_llm_response("standard")
                for comp_name, component in reward_components.items():
                    score = component.calculate(state)
                    normalizer.normalize(score, comp_name) # Adapt step
                total_score = RewardComposer(reward_components).calculate(state, None)
                normalizer.normalize(total_score, "total")


        results, _ = evaluate_response_types(reward_components, normalizer)
        all_results[norm_name] = [results[rt] for rt in response_types]
    
    fig, ax = plt.subplots(figsize=(14, 7)) # Increased figure size
    
    bar_width = 0.8 / len(normalizers)
    x_indices = np.arange(len(response_types))

    for i, (norm_name, values) in enumerate(all_results.items()):
        x_positions = x_indices + (i - len(normalizers)/2 + 0.5) * bar_width
        ax.bar(x_positions, values, width=bar_width, label=norm_name, alpha=0.7)
    
    ax.set_title('Effect of Different Normalization Methods on Total Reward')
    ax.set_ylabel('Normalized Reward')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(response_types, rotation=45, ha="right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(normalizers)) # Adjusted legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Normalization comparison visualization saved to {save_path}")
    
    return save_path

def main():
    # Create reward components
    factuality = FactualityReward(weight=1.5, threshold=0.6)
    coherence = CoherenceReward(weight=1.0)
    relevance = RelevanceReward(weight=1.2, query_importance=0.6)
    helpfulness = HelpfulnessReward(weight=1.0)
    harmlessness = HarmlessnessReward(weight=2.0, toxicity_penalty=5.0)
    concision = ConcisionReward(weight=0.8, target_length=250, tolerance=100)
    diversity = DiversityReward(weight=0.7)
    grounding = GroundingReward(weight=1.0, citation_bonus=0.2)
    alignment = AlignmentReward(weight=1.3) # Assuming some alignment score is available
    
    # Combine components
    reward_components = {
        "factuality": factuality,
        "coherence": coherence,
        "relevance": relevance,
        "helpfulness": helpfulness,
        "harmlessness": harmlessness,
        "concision": concision,
        "diversity": diversity,
        "grounding": grounding,
        "alignment": alignment
    }
    
    print("--- Evaluating without Normalization ---")
    results_no_norm, component_results_no_norm = evaluate_response_types(reward_components)
    visualize_rewards(results_no_norm, component_results_no_norm, "llm_rewards_no_norm.png")
    
    print("\n--- Evaluating with Standard Normalization ---")
    standard_normalizer = RewardNormalizer(method="standard")
    # Warm up the normalizer
    for _ in range(100): # Simulate 100 steps
        state = simulate_llm_response("standard")
        for comp_name, component in reward_components.items():
            score = component.calculate(state)
            standard_normalizer.normalize(score, comp_name) # Adapt step
        total_score = RewardComposer(reward_components).calculate(state, None)
        standard_normalizer.normalize(total_score, "total")

    results_std_norm, component_results_std_norm = evaluate_response_types(reward_components, standard_normalizer)
    visualize_rewards(results_std_norm, component_results_std_norm, "llm_rewards_std_norm.png")

    print("\n--- Comparing Normalization Methods ---")
    compare_normalization_methods(reward_components, "llm_normalization_comparison.png")

    print("\nExample finished. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()