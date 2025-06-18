import matplotlib.pyplot as plt
import numpy as np
import optuna
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

class RewardOptimizationVisualizer:
    def __init__(self, study: optuna.study.Study):
        self.study = study
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> Figure:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        trials = self.study.trials
        values = [t.value for t in trials if t.value is not None]
        iterations = list(range(len(values)))
        
        best_values = [max(values[:i+1]) for i in range(len(values))]
        
        ax.plot(iterations, values, 'o-', label='Trial Value')
        ax.plot(iterations, best_values, 'r-', label='Best Value')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_param_importances(self, save_path: Optional[str] = None) -> Figure:
        importances = optuna.importance.get_param_importances(self.study)
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        params = list(importances.keys())
        values = list(importances.values())
        
        y_pos = np.arange(len(params))
        
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Parameter Importances')
        ax.grid(True, axis='x')
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_parallel_coordinate(self, save_path: Optional[str] = None) -> Figure:
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_contour(self, params: Optional[List[str]] = None, save_path: Optional[str] = None) -> Figure:
        fig = optuna.visualization.matplotlib.plot_contour(self.study, params=params)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_slice(self, params: Optional[List[str]] = None, save_path: Optional[str] = None) -> Figure:
        fig = optuna.visualization.matplotlib.plot_slice(self.study, params=params)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_edf(self, save_path: Optional[str] = None) -> Figure:
        fig = optuna.visualization.matplotlib.plot_edf(self.study)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def create_dashboard(self, save_path: str = "optimization_dashboard.png"):
        fig = plt.figure(figsize=(20, 15))
        
        plt.subplot(2, 2, 1)
        optuna.visualization.matplotlib.plot_optimization_history(self.study)
        plt.title("Optimization History")
        
        plt.subplot(2, 2, 2)
        optuna.visualization.matplotlib.plot_param_importances(self.study)
        plt.title("Parameter Importances")
        
        plt.subplot(2, 2, 3)
        optuna.visualization.matplotlib.plot_slice(self.study)
        plt.title("Slice Plot")
        
        plt.subplot(2, 2, 4)
        optuna.visualization.matplotlib.plot_contour(self.study)
        plt.title("Contour Plot")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        return save_path

class RewardComponentVisualizer:
    def __init__(self):
        self.reward_history = {}
        self.component_history = {}
    
    def record_reward(self, total_reward: float, component_rewards: Dict[str, float], step: int):
        self.reward_history[step] = total_reward
        
        for component, value in component_rewards.items():
            if component not in self.component_history:
                self.component_history[component] = {}
            self.component_history[component][step] = value
    
    def plot_reward_history(self, window_size: int = 1, save_path: Optional[str] = None) -> Figure:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        steps = sorted(self.reward_history.keys())
        rewards = [self.reward_history[step] for step in steps]
        
        if window_size > 1:
            smoothed_rewards = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                smoothed_rewards.append(np.mean(rewards[start_idx:i+1]))
            ax.plot(steps, smoothed_rewards, 'b-', label=f'Smoothed (window={window_size})')
        
        ax.plot(steps, rewards, 'k-', alpha=0.3, label='Raw')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward History')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_component_contributions(self, save_path: Optional[str] = None) -> Figure:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        steps = sorted(self.reward_history.keys())
        
        component_data = {}
        for component, history in self.component_history.items():
            component_data[component] = [history.get(step, 0.0) for step in steps]
        
        df = pd.DataFrame(component_data, index=steps)
        df.plot.area(ax=ax, stacked=True, alpha=0.7)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward Contribution')
        ax.set_title('Component Contributions Over Time')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_component_correlation(self, save_path: Optional[str] = None) -> Figure:
        components = list(self.component_history.keys())
        if len(components) < 2:
            return None
        
        steps = sorted(self.reward_history.keys())
        
        data = {}
        data['total'] = [self.reward_history[step] for step in steps]
        
        for component, history in self.component_history.items():
            data[component] = [history.get(step, 0.0) for step in steps]
        
        df = pd.DataFrame(data)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Between Reward Components')
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def plot_component_distribution(self, save_path: Optional[str] = None) -> Figure:
        components = list(self.component_history.keys())
        if not components:
            return None
        
        fig = plt.figure(figsize=(12, 6))
        
        for i, component in enumerate(components):
            values = list(self.component_history[component].values())
            
            ax = fig.add_subplot(1, len(components), i+1)
            sns.histplot(values, kde=True, ax=ax)
            ax.set_title(f'{component} Distribution')
            ax.set_xlabel('Reward Value')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
            return None
        
        return fig
    
    def create_dashboard(self, save_path: str = "reward_dashboard.png"):
        fig = plt.figure(figsize=(20, 15))
        
        plt.subplot(2, 2, 1)
        steps = sorted(self.reward_history.keys())
        rewards = [self.reward_history[step] for step in steps]
        plt.plot(steps, rewards, 'b-')
        plt.title("Reward History")
        plt.xlabel("Step")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        component_data = {}
        for component, history in self.component_history.items():
            component_data[component] = [history.get(step, 0.0) for step in steps]
        df = pd.DataFrame(component_data, index=steps)
        df.plot.area(stacked=True, alpha=0.7)
        plt.title("Component Contributions")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        data = {}
        data['total'] = [self.reward_history[step] for step in steps]
        for component, history in self.component_history.items():
            data[component] = [history.get(step, 0.0) for step in steps]
        df = pd.DataFrame(data)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title("Component Correlations")
        
        plt.subplot(2, 2, 4)
        components = list(self.component_history.keys())
        for component in components:
            values = list(self.component_history[component].values())
            plt.hist(values, alpha=0.5, label=component)
        plt.title("Component Distributions")
        plt.xlabel("Reward Value")
        plt.ylabel("Frequency")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        return save_path