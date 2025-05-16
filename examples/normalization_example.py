import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllama.rewards.normalization import RewardNormalizer, PopArtNormalizer, AdaptiveNormalizer

def generate_non_stationary_rewards(n_steps: int = 1000) -> List[float]:
    rewards = []
    
    for i in range(n_steps):
        if i < n_steps // 3:
            base = 1.0
            scale = 0.5
        elif i < 2 * n_steps // 3:
            base = 5.0
            scale = 2.0
        else:
            base = -2.0
            scale = 1.0
        
        reward = base + scale * np.random.randn()
        rewards.append(reward)
    
    return rewards

def visualize_normalization(rewards: List[float], normalizers: Dict[str, RewardNormalizer], save_path: str = "normalization_comparison.png"):
    normalized_rewards = {}
    
    for name, normalizer in normalizers.items():
        normalized = []
        for r in rewards:
            normalized.append(normalizer.normalize(r))
        normalized_rewards[name] = normalized
    
    fig, axes = plt.subplots(len(normalizers) + 1, 1, figsize=(12, 3 * (len(normalizers) + 1)), sharex=True)
    
    axes[0].plot(rewards)
    axes[0].set_title("Original Rewards")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)
    
    for i, (name, norm_rewards) in enumerate(normalized_rewards.items(), 1):
        axes[i].plot(norm_rewards)
        axes[i].set_title(f"Normalized with {name}")
        axes[i].set_ylabel("Normalized Reward")
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Step")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path

def compare_normalizer_stability(rewards: List[float], save_path: str = "normalizer_stability.png"):
    standard_norm = RewardNormalizer(method="standard")
    popart_norm = PopArtNormalizer(beta=0.001)
    adaptive_norm = AdaptiveNormalizer()
    
    standard_normalized = []
    popart_normalized = []
    adaptive_normalized = []
    
    standard_stats = []
    popart_stats = []
    
    for r in rewards:
        standard_normalized.append(standard_norm.normalize(r))
        popart_normalized.append(popart_norm.normalize(r))
        adaptive_normalized.append(adaptive_norm.normalize(r))
        
        standard_stats.append((standard_norm.stats["default"]["mean"], standard_norm.stats["default"]["std"]))
        popart_stats.append((popart_norm.mean, popart_norm.std))
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    axes[0].plot(rewards, 'k-', alpha=0.7, label='Original')
    axes[0].set_title("Original Rewards")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)
    
    axes[1].plot(standard_normalized, 'b-', label='Standard')
    axes[1].plot(popart_normalized, 'r-', label='PopArt')
    axes[1].plot(adaptive_normalized, 'g-', label='Adaptive')
    axes[1].set_title("Normalized Rewards Comparison")
    axes[1].set_ylabel("Normalized Reward")
    axes[1].legend()
    axes[1].grid(True)
    
    standard_means, standard_stds = zip(*standard_stats)
    popart_means, popart_stds = zip(*popart_stats)
    
    axes[2].plot(standard_means, 'b-', label='Standard Mean')
    axes[2].plot(standard_stds, 'b--', label='Standard Std')
    axes[2].plot(popart_means, 'r-', label='PopArt Mean')
    axes[2].plot(popart_stds, 'r--', label='PopArt Std')
    axes[2].set_title("Normalizer Statistics")
    axes[2].set_ylabel("Value")
    axes[2].set_xlabel("Step")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path

def demonstrate_adaptive_phases(n_steps: int = 5000, save_path: str = "adaptive_phases.png"):
    rewards = []
    
    for i in range(n_steps):
        if i < n_steps // 5:
            base = 0.0
            scale = 1.0
        elif i < 2 * n_steps // 5:
            base = 10.0
            scale = 3.0
        elif i < 3 * n_steps // 5:
            base = -5.0
            scale = 2.0
        elif i < 4 * n_steps // 5:
            base = 2.0
            scale = 0.5
        else:
            base = 0.0
            scale = 0.2
        
        reward = base + scale * np.random.randn()
        rewards.append(reward)
    
    phase_thresholds = {
        "exploration": n_steps // 5,
        "learning": 3 * n_steps // 5,
        "exploitation": 4 * n_steps // 5
    }
    
    adaptive_norm = AdaptiveNormalizer(phase_thresholds=phase_thresholds)
    
    normalized = []
    phases = []
    
    for r in rewards:
        normalized.append(adaptive_norm.normalize(r))
        phases.append(adaptive_norm.current_phase)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(rewards, 'k-', alpha=0.7)
    axes[0].set_title("Original Rewards")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True)
    
    axes[1].plot(normalized, 'g-')
    
    phase_colors = {
        "exploration": "blue",
        "learning": "green",
        "exploitation": "red"
    }
    
    for phase, color in phase_colors.items():
        phase_indices = [i for i, p in enumerate(phases) if p == phase]
        if phase_indices:
            axes[1].axvspan(min(phase_indices), max(phase_indices), alpha=0.2, color=color, label=f"{phase} phase")
    
    axes[1].set_title("Adaptive Normalized Rewards")
    axes[1].set_ylabel("Normalized Reward")
    axes[1].set_xlabel("Step")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    return save_path

def main():
    print("Generating non-stationary rewards...")
    rewards = generate_non_stationary_rewards(1000)
    
    normalizers = {
        "Standard": RewardNormalizer(method="standard"),
        "MinMax": RewardNormalizer(method="minmax"),
        "Robust": RewardNormalizer(method="robust"),
        "Tanh": RewardNormalizer(method="tanh", scale=2.0),
        "PopArt": PopArtNormalizer(beta=0.001)
    }
    
    print("Visualizing normalization methods...")
    vis_path = visualize_normalization(rewards, normalizers)
    print(f"Visualization saved to {vis_path}")
    
    print("Comparing normalizer stability...")
    stability_path = compare_normalizer_stability(rewards)
    print(f"Stability comparison saved to {stability_path}")
    
    print("Demonstrating adaptive normalization phases...")
    adaptive_path = demonstrate_adaptive_phases()
    print(f"Adaptive phases visualization saved to {adaptive_path}")
    
    print("All visualizations complete!")

if __name__ == "__main__":
    main()