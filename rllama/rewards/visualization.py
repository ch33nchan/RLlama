import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional

class RewardDashboard:
    """
    Logs reward component data and generates an interactive dashboard.
    """
    def __init__(self):
        self.log_data: List[Dict] = []
        self.steps: List[int] = []

    def log_iteration(self, weights: Dict[str, float], metrics: Dict[str, float], step: int):
        """
        Logs the weights and raw reward values for a single step.

        Args:
            weights: Dictionary mapping reward component names to their current weights.
            metrics: Dictionary mapping reward component names to their raw values.
            step: The current global step number.
        """
        log_entry = {'step': step}
        # Prefix keys to distinguish weights and metrics
        for name, weight in weights.items():
            log_entry[f"weight_{name}"] = weight
        for name, metric in metrics.items():
            log_entry[f"metric_{name}"] = metric
        self.log_data.append(log_entry)
        # self.steps.append(step) # Redundant if step is in log_entry

    def generate_dashboard(self, output_file: str = "reward_dashboard.html"):
        """
        Generates an interactive HTML dashboard using Plotly.
        """
        if not self.log_data:
            print("No data logged to generate dashboard.")
            return

        df = pd.DataFrame(self.log_data)
        df = df.sort_values(by='step').set_index('step')

        if df.empty:
            print("DataFrame is empty after processing log data.")
            return

        weight_cols = [col for col in df.columns if col.startswith('weight_')]
        metric_cols = [col for col in df.columns if col.startswith('metric_')]

        if not weight_cols and not metric_cols:
            print("No weight or metric columns found in the data.")
            return

        num_plots = (1 if weight_cols else 0) + (1 if metric_cols else 0)
        if num_plots == 0:
             print("No data to plot.")
             return

        fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True,
                            subplot_titles=("Reward Weights Over Time", "Raw Reward Metrics Over Time") if num_plots==2 else ("Reward Weights/Metrics Over Time",))

        current_row = 1
        # Plot Weights
        if weight_cols:
            for col in weight_cols:
                component_name = col.replace('weight_', '')
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f"Weight: {component_name}"),
                              row=current_row, col=1)
            current_row += 1

        # Plot Metrics
        if metric_cols:
             for col in metric_cols:
                 component_name = col.replace('metric_', '')
                 fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f"Metric: {component_name}"),
                               row=current_row, col=1)


        fig.update_layout(
            title_text="Reward Shaping Analysis",
            hovermode="x unified",
            height=300 * num_plots # Adjust height based on number of plots
        )
        fig.update_xaxes(title_text="Global Step")
        # Update y-axis titles if needed
        if weight_cols:
             fig.update_yaxes(title_text="Weight Value", row=1, col=1)
        if metric_cols:
             fig.update_yaxes(title_text="Raw Reward Value", row=current_row if weight_cols else 1, col=1)


        try:
            fig.write_html(output_file)
            print(f"Dashboard generated: {output_file}")
        except Exception as e:
            print(f"Error writing dashboard HTML file: {e}")