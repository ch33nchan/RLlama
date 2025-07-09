from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(
    data: Dict[str, List[Tuple[int, float]]],
    x_key: str = "steps",
    y_keys: List[str] = ["episode_reward"],
    window_size: int = 10,
    title: str = "Learning Curve",
    xlabel: str = "Steps",
    ylabel: str = "Reward",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot learning curves from logged metrics.
    
    Args:
        data: Dictionary of metrics, where keys are metric names and values are
            lists of (step, value) tuples
        x_key: Key for x-axis values
        y_keys: Keys for y-axis values
        window_size: Window size for smoothing
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract x values if provided
    if x_key in data:
        x_values = [x for x, _ in data[x_key]]
    else:
        # Use step indices as x values
        x_values = [i for i in range(len(data[y_keys[0]]))]
        
    # Plot each y key
    for key in y_keys:
        if key not in data:
            continue
            
        # Extract values
        values = [v for _, v in data[key]]
        
        # Apply smoothing
        if window_size > 1:
            smoothed_values = np.convolve(values, np.ones(window_size) / window_size, mode="valid")
            
            # Adjust x values for the smoothed curve
            smoothed_x = x_values[window_size-1:]
            
            # Plot both raw and smoothed curves
            ax.plot(x_values, values, alpha=0.3, label=f"{key} (raw)")
            ax.plot(smoothed_x, smoothed_values, label=f"{key} (smoothed)")
        else:
            # Just plot raw values
            ax.plot(x_values, values, label=key)
            
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig


def plot_episode_stats(
    episode_lengths: List[int],
    episode_rewards: List[float],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot episode statistics.
    
    Args:
        episode_lengths: List of episode lengths
        episode_rewards: List of episode rewards
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot episode lengths
    axes[0, 0].plot(episode_lengths)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Length")
    axes[0, 0].set_title("Episode Lengths")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode rewards
    axes[0, 1].plot(episode_rewards)
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_title("Episode Rewards")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot episode length histogram
    axes[1, 0].hist(episode_lengths, bins=20)
    axes[1, 0].set_xlabel("Length")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Episode Length Distribution")
    
    # Plot episode reward histogram
    axes[1, 1].hist(episode_rewards, bins=20)
    axes[1, 1].set_xlabel("Reward")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Episode Reward Distribution")
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig


def plot_action_distribution(
    actions: List[Any],
    discrete: bool = True,
    num_bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot distribution of actions.
    
    Args:
        actions: List of actions
        discrete: Whether actions are discrete
        num_bins: Number of bins for histogram
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if discrete:
        # Count frequency of each action
        action_counts = {}
        for action in actions:
            if isinstance(action, np.ndarray) and action.size == 1:
                action = action.item()
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
            
        # Plot bar chart
        action_ids = sorted(action_counts.keys())
        frequencies = [action_counts[a] for a in action_ids]
        
        ax.bar(action_ids, frequencies)
        ax.set_xlabel("Action")
        ax.set_ylabel("Frequency")
        ax.set_title("Discrete Action Distribution")
        
        # Set integer ticks for discrete actions
        ax.set_xticks(action_ids)
    else:
        # Convert actions to array
        actions_array = np.array(actions)
        
        # Handle multidimensional actions
        if len(actions_array.shape) > 1:
            # Plot histogram for each dimension
            for i in range(actions_array.shape[1]):
                ax.hist(
                    actions_array[:, i],
                    bins=num_bins,
                    alpha=0.5,
                    label=f"Dimension {i}"
                )
            ax.legend()
        else:
            # Plot single histogram
            ax.hist(actions_array, bins=num_bins)
            
        ax.set_xlabel("Action Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Continuous Action Distribution")
        
    ax.grid(True, alpha=0.3)
    
    # Save figure if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
    # Show figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return fig