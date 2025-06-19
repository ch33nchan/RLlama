

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any, Union
import time
from matplotlib.figure import Figure
from collections import defaultdict

class RewardVisualizer:
    """
    Visualizer for reward data collected by RLlama.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            log_dir: Directory containing reward logs
        """
        self.log_dir = log_dir
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load reward data from log files.
        
        Returns:
            Dictionary mapping file names to lists of reward data
        """
        data = {}
        
        # Find all log files
        if not os.path.exists(self.log_dir):
            return data
            
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.log_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        file_data = json.load(f)
                        
                    data[filename] = file_data
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    
        return data
        
    def plot_total_rewards(self, 
                          smoothing: int = 1, 
                          figsize: tuple = (12, 6)) -> Figure:
        """
        Plot total rewards over time.
        
        Args:
            smoothing: Window size for smoothing rewards
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        for filename, file_data in self.data.items():
            if not file_data:
                continue
                
            # Extract steps and rewards
            steps = [entry.get('step', i) for i, entry in enumerate(file_data)]
            rewards = [entry.get('total_reward', 0) for entry in file_data]
            
            # Apply smoothing if specified
            if smoothing > 1 and len(rewards) > smoothing:
                smooth_rewards = []
                for i in range(len(rewards) - smoothing + 1):
                    smooth_rewards.append(np.mean(rewards[i:i+smoothing]))
                rewards = smooth_rewards
                steps = steps[:len(smooth_rewards)]
                
            # Plot
            plt.plot(steps, rewards, label=filename)
            
        plt.xlabel('Step')
        plt.ylabel('Total Reward')
        plt.title('Total Rewards Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
        
    def plot_component_rewards(self, 
                              component_names: Optional[List[str]] = None,
                              figsize: tuple = (15, 8)) -> Figure:
        """
        Plot component-wise rewards over time.
        
        Args:
            component_names: Names of components to plot (all if None)
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Collect all component names if not specified
        all_components = set()
        for file_data in self.data.values():
            for entry in file_data:
                components = entry.get('component_rewards', {})
                all_components.update(components.keys())
                
        if not component_names:
            component_names = sorted(all_components)
            
        # Create plot
        plt.figure(figsize=figsize)
        
        # We'll use subplots for different components
        num_components = len(component_names)
        if num_components == 0:
            return plt.gcf()
            
        # Calculate grid dimensions
        cols = min(3, num_components)
        rows = (num_components + cols - 1) // cols
        
        # Plot each component
        for i, component in enumerate(component_names):
            plt.subplot(rows, cols, i + 1)
            
            for filename, file_data in self.data.items():
                steps = []
                rewards = []
                
                for j, entry in enumerate(file_data):
                    component_rewards = entry.get('component_rewards', {})
                    if component in component_rewards:
                        steps.append(entry.get('step', j))
                        rewards.append(component_rewards[component])
                        
                if steps and rewards:
                    plt.plot(steps, rewards, label=filename)
                    
            plt.title(component)
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.legend()
        
        return plt.gcf()
        
    def component_contribution(self, figsize: tuple = (10, 6)) -> Figure:
        """
        Plot pie chart showing the average contribution of each component.
        
        Args:
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        # Calculate total contribution per component
        component_totals = defaultdict(float)
        component_counts = defaultdict(int)
        
        for file_data in self.data.values():
            for entry in file_data:
                components = entry.get('component_rewards', {})
                for component, reward in components.items():
                    component_totals[component] += abs(reward)  # Use absolute values
                    component_counts[component] += 1
                    
        # Calculate average contribution
        component_avgs = {}
        for component, total in component_totals.items():
            count = component_counts[component]
            if count > 0:
                component_avgs[component] = total / count
                
        # Create pie chart
        plt.figure(figsize=figsize)
        
        if component_avgs:
            labels = list(component_avgs.keys())
            values = list(component_avgs.values())
            
            plt.pie(values, labels=labels, autopct='%1.1f%%')
            plt.axis('equal')
            plt.title('Average Absolute Contribution by Component')
            
        return plt.gcf()
        
    def save_plots(self, output_dir: str) -> None:
        """
        Save all plots to the specified directory.
        
        Args:
            output_dir: Directory to save plots to
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save total rewards plot
        fig = self.plot_total_rewards()
        fig.savefig(os.path.join(output_dir, 'total_rewards.png'))
        plt.close(fig)
        
        # Save component rewards plot
        fig = self.plot_component_rewards()
        fig.savefig(os.path.join(output_dir, 'component_rewards.png'))
        plt.close(fig)
        
        # Save component contribution plot
        fig = self.component_contribution()
        fig.savefig(os.path.join(output_dir, 'component_contribution.png'))
        plt.close(fig)
        
        print(f"Plots saved to {output_dir}")
