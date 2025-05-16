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
        total_reward = composer.calculate(state, None)
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
    axes[0].set_xticklabels(response_types, rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot component rewards
    bar_width = 0.8 / len(component_names)
    for i, component in enumerate(component_names):
        component_values = [component_results[rt][component] for rt in response_types]
        x_positions = np.arange(len(response_types)) + (i - len(component_names)/2 + 0.5) * bar_width
        axes[1].bar(x_positions, component_values, width=bar_width, 
                   label=component, alpha=0.7)
    
    axes[1].set_title('Component Rewards by Response Type')
    axes[1].set_ylabel('Component Reward')
    axes[1].set_xticks(np.arange(len(response_types)))
    axes[1].set_xticklabels(response_types, rotation=45)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
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
        "Adaptive": AdaptiveNormalizer()
    }
    
    all_results = {}
    
    for norm_name, normalizer in normalizers.items():
        results, _ = evaluate_response_types(reward_components, normalizer)
        all_results[norm_name] = [results[rt] for rt in response_types]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.8 / len(normalizers)
    for i, (norm_name, values) in enumerate(all_results.items()):
        x_positions = np.arange(len(response_types)) + (i - len(normalizers)/2 + 0.5) * bar_width
        ax.bar(x_positions, values, width=bar_width, label=norm_name, alpha=0.7)
    
    ax.set_title('Effect of Different Normalization Methods')
    ax.set_ylabel('Normalized Reward')
    ax.set_xticks(np.arange(len(response_types)))
    ax.set_xticklabels(response_types, rotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(normalizers))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
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
    alignment = AlignmentReward(weight=1.3)
    
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
    
    # Create normalizer
    normalizer = AdaptiveNormalizer()
    
    print("Evaluating different response types...")
    results, component_results = evaluate_response_types(reward_components, normalizer)
    
    print("\nResults:")
    for response_type, reward in results.items():
        print(f"  {response_type}: {reward:.4f}")
    
    print("\nVisualizing rewards...")
    vis_path = visualize_rewards(results, component_results)
    print(f"Visualization saved to {vis_path}")
    
    print("\nComparing normalization methods...")
    norm_path = compare_normalization_methods(reward_components)
    print(f"Normalization comparison saved to {norm_path}")
    
    # Test different composition methods
    print("\nTesting different composition methods...")
    composer = RewardComposer(reward_components)
    
    composition_methods = [
        "linear", "multiplicative", "min", "max", 
        "geometric_mean", "softmax"
    ]
    
    composition_results = {}
    
    for method in composition_methods:
        if method == "softmax":
            composer.set_composition_method(method, temperature=0.5)
        elif method == "geometric_mean":
            composer.set_composition_method(method, offset=1.0)
        else:
            composer.set_composition_method(method)
        
        method_results = {}
        for response_type in ["standard", "factual_error", "toxic", "perfect"]:
            state = simulate_llm_response(response_type)
            reward = composer.calculate(state, None)
            method_results[response_type] = reward
        
        composition_results[method] = method_results
    
    # Visualize composition method comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    response_types = ["standard", "factual_error", "toxic", "perfect"]
    bar_width = 0.8 / len(composition_methods)
    
    for i, method in enumerate(composition_methods):
        values = [composition_results[method][rt] for rt in response_types]
        x_positions = np.arange(len(response_types)) + (i - len(composition_methods)/2 + 0.5) * bar_width
        ax.bar(x_positions, values, width=bar_width, label=method, alpha=0.7)
    
    ax.set_title('Effect of Different Composition Methods')
    ax.set_ylabel('Reward')
    ax.set_xticks(np.arange(len(response_types)))
    ax.set_xticklabels(response_types, rotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("composition_comparison.png")
    plt.close(fig)
    
    print("Composition method comparison saved to composition_comparison.png")
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main()