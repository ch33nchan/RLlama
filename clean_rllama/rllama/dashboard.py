# rllama/dashboard.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

def launch_dashboard(log_dir: str = "./rllama_logs", port: int = 8501):
    """
    Launch the RLlama dashboard with Streamlit.
    
    Args:
        log_dir: Directory containing logs to visualize.
        port: Port to run the dashboard on.
    """
    import subprocess
    import sys
    
    # Find the dashboard script path
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "streamlit_app.py")
    
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard not found at {dashboard_path}")
        return
    
    print("🦙 Launching RLlama Dashboard...")
    print(f"Dashboard will open in your web browser at http://localhost:{port}")
    print("Press Ctrl+C to stop the dashboard")
    
    # Pass log_dir as an environment variable
    env = os.environ.copy()
    env["RLLAMA_LOG_DIR"] = log_dir
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", str(port),
            "--server.address", "localhost"
        ], env=env)
    except KeyboardInterrupt:
        print("\n🦙 Dashboard stopped.")


class RewardTracker:
    """
    Track and visualize reward components during training.
    """
    
    def __init__(self, log_dir: str = "./rllama_reward_logs"):
        """
        Initialize the reward tracker.
        
        Args:
            log_dir: Directory to save reward logs.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.reward_history = {}
        self.step_counter = 0
        self.episode_counter = 0
        
        # Track when the episode starts
        self.episode_starts = [0]
    
    def log_rewards(self, 
                    component_rewards: Dict[str, float], 
                    final_reward: float, 
                    metadata: Dict[str, Any] = None):
        """
        Log reward values for the current step.
        
        Args:
            component_rewards: Dictionary of component name to reward value.
            final_reward: The final combined reward value.
            metadata: Additional metadata to log.
        """
        # Initialize history for new components
        for component in component_rewards:
            if component not in self.reward_history:
                self.reward_history[component] = []
        
        # Add special "final" component if it doesn't exist
        if "final_reward" not in self.reward_history:
            self.reward_history["final_reward"] = []
        
        # Log all components
        for component, value in component_rewards.items():
            # Handle NaN values
            if isinstance(value, float) and np.isnan(value):
                value = 0.0
                
            self.reward_history[component].append(value)
        
        # Log final reward
        self.reward_history["final_reward"].append(final_reward)
        
        # Update step counter
        self.step_counter += 1
    
    def end_episode(self):
        """Mark the end of an episode for visualization"""
        self.episode_counter += 1
        self.episode_starts.append(self.step_counter)
        
    def save(self, filename: str = None):
        """
        Save reward history to a file.
        
        Args:
            filename: Name of the file to save to (without extension).
                     If None, uses timestamp.
        """
        if filename is None:
            filename = f"reward_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        # Convert to pandas DataFrame
        all_rewards = {}
        max_len = 0
        
        for component, values in self.reward_history.items():
            all_rewards[component] = values
            max_len = max(max_len, len(values))
        
        # Pad shorter arrays with NaN
        for component, values in all_rewards.items():
            if len(values) < max_len:
                all_rewards[component] = values + [float('nan')] * (max_len - len(values))
                
        # Create DataFrame
        df = pd.DataFrame(all_rewards)
        
        # Add step column
        df['step'] = range(len(df))
        
        # Add episode column
        episode_col = np.zeros(len(df), dtype=int)
        for i in range(len(self.episode_starts) - 1):
            start_idx = self.episode_starts[i]
            end_idx = self.episode_starts[i + 1]
            episode_col[start_idx:end_idx] = i
        
        # Fill the last episode
        if len(self.episode_starts) > 0:
            episode_col[self.episode_starts[-1]:] = len(self.episode_starts) - 1
            
        df['episode'] = episode_col
        
        # Save as CSV
        filepath = os.path.join(self.log_dir, f"{filename}.csv")
        df.to_csv(filepath, index=False)
        
        # Also save metadata
        metadata = {
            "total_steps": self.step_counter,
            "total_episodes": self.episode_counter,
            "components": list(self.reward_history.keys()),
            "episode_starts": self.episode_starts,
            "timestamp": datetime.now().isoformat()
        }
        
        meta_filepath = os.path.join(self.log_dir, f"{filename}_meta.json")
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return filepath
    
    @staticmethod
    def load(filepath: str) -> 'RewardTracker':
        """
        Load reward history from a saved file.
        
        Args:
            filepath: Path to the CSV file to load.
            
        Returns:
            A RewardTracker instance with the loaded data.
        """
        if not filepath.endswith('.csv'):
            filepath = filepath + '.csv'
            
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Create new tracker instance
        tracker = RewardTracker(log_dir=os.path.dirname(filepath))
        
        # Load data into tracker
        components = [col for col in df.columns if col not in ['step', 'episode']]
        
        for component in components:
            tracker.reward_history[component] = df[component].fillna(0).tolist()
            
        tracker.step_counter = len(df)
        
        # Try to load metadata if available
        meta_filepath = filepath.replace('.csv', '_meta.json')
        if os.path.exists(meta_filepath):
            with open(meta_filepath, 'r') as f:
                metadata = json.load(f)
                tracker.episode_counter = metadata.get("total_episodes", 0)
                tracker.episode_starts = metadata.get("episode_starts", [0])
        else:
            # Extract episodes from the dataframe
            if 'episode' in df.columns:
                tracker.episode_counter = df['episode'].max() + 1
                
                # Reconstruct episode_starts
                episode_changes = df['episode'].diff().fillna(0)
                start_indices = df.index[episode_changes > 0].tolist()
                tracker.episode_starts = [0] + start_indices
                
        return tracker
    
    def plot_rewards(self, 
                     components: List[str] = None, 
                     window_size: int = 10,
                     show_episodes: bool = True,
                     figsize: tuple = (12, 6),
                     save_path: Optional[str] = None):
        """
        Plot reward history with smoothing.
        
        Args:
            components: List of component names to plot. If None, plots all.
            window_size: Window size for smoothing.
            show_episodes: Whether to show episode boundaries.
            figsize: Figure size.
            save_path: Path to save the figure to.
            
        Returns:
            The matplotlib figure.
        """
        import matplotlib.pyplot as plt
        
        if components is None:
            components = [c for c in self.reward_history.keys()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each component
        for component in components:
            if component in self.reward_history:
                values = self.reward_history[component]
                steps = range(len(values))
                
                # Apply smoothing if requested
                if window_size > 1:
                    values = pd.Series(values).rolling(window=window_size).mean().fillna(0).values
                    
                ax.plot(steps, values, label=component)
        
        # Add episode markers
        if show_episodes and len(self.episode_starts) > 1:
            for start in self.episode_starts[1:]:
                ax.axvline(x=start, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels and legend
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Components')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path)
        
        return fig


class RewardVisualizer:
    """
    Visualize reward data from logs using Plotly for interactive charts.
    """
    
    def __init__(self, log_dir: str = "./rllama_reward_logs"):
        """
        Initialize the visualizer.
        
        Args:
            log_dir: Directory containing reward logs.
        """
        self.log_dir = log_dir
        self.trackers = {}
        self.load_logs()
        
    def load_logs(self):
        """Load all reward logs from the log directory"""
        log_dir = Path(self.log_dir)
        if not log_dir.exists():
            return
        
        # Find all CSV files
        csv_files = list(log_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            # Skip meta files
            if "_meta" in csv_file.stem:
                continue
                
            try:
                tracker = RewardTracker.load(str(csv_file))
                self.trackers[csv_file.stem] = tracker
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
    
    def plot_interactive(self, 
                        log_name: str, 
                        components: List[str] = None,
                        window_size: int = 10,
                        show_episodes: bool = True):
        """
        Create an interactive plot using Plotly.
        
        Args:
            log_name: Name of the log file to visualize.
            components: Components to include in the plot.
            window_size: Window size for smoothing.
            show_episodes: Whether to show episode boundaries.
            
        Returns:
            A Plotly figure.
        """
        if log_name not in self.trackers:
            print(f"Log {log_name} not found.")
            return None
        
        tracker = self.trackers[log_name]
        
        # Determine components to plot
        if components is None:
            components = list(tracker.reward_history.keys())
        else:
            # Filter to only include components that exist
            components = [c for c in components if c in tracker.reward_history]
        
        # Create a DataFrame for plotting
        plot_data = {}
        for component in components:
            values = tracker.reward_history[component]
            
            # Apply smoothing
            if window_size > 1:
                values = pd.Series(values).rolling(window=window_size).mean().fillna(0).values
            
            plot_data[component] = values
            
        plot_data['step'] = range(len(next(iter(plot_data.values()))))
        df = pd.DataFrame(plot_data)
        
        # Reshape for plotly
        df_melted = pd.melt(df, id_vars=['step'], value_vars=components, 
                          var_name='Component', value_name='Reward')
        
        # Create plotly figure
        fig = px.line(df_melted, x='step', y='Reward', color='Component',
                    title=f"Reward Components - {log_name}")
        
        # Add episode boundaries
        if show_episodes and len(tracker.episode_starts) > 1:
            for start in tracker.episode_starts[1:]:
                fig.add_shape(
                    type="line", line=dict(dash="dash", color="gray"),
                    x0=start, y0=0, x1=start, y1=1, yref="paper"
                )
        
        # Update layout
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Step",
            yaxis_title="Reward Value",
            template="plotly_white"
        )
        
        return fig
    
    def plot_component_comparison(self,
                                 log_names: List[str],
                                 component: str,
                                 window_size: int = 10,
                                 title: str = None):
        """
        Compare a specific component across multiple logs.
        
        Args:
            log_names: List of log names to compare.
            component: The component to compare.
            window_size: Window size for smoothing.
            title: Plot title.
            
        Returns:
            A Plotly figure.
        """
        # Check if the requested logs exist
        valid_logs = [log for log in log_names if log in self.trackers]
        
        if not valid_logs:
            print("No valid logs found.")
            return None
        
        # Create plot data
        data = []
        
        for log_name in valid_logs:
            tracker = self.trackers[log_name]
            
            if component in tracker.reward_history:
                values = tracker.reward_history[component]
                
                # Apply smoothing
                if window_size > 1:
                    values = pd.Series(values).rolling(window=window_size).mean().fillna(0).values
                    
                data.append(go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines',
                    name=log_name
                ))
        
        # Create figure
        fig = go.Figure(data=data)
        
        # Update layout
        fig.update_layout(
            title=title or f"Comparison of {component} Across Runs",
            xaxis_title="Step",
            yaxis_title=f"{component} Value",
            hovermode="x unified",
            template="plotly_white"
        )
        
        return fig