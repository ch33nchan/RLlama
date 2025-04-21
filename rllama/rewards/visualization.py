import plotly.graph_objects as go
import pandas as pd

class RewardDashboard:
    def __init__(self):
        self.history = []
        
    def log_iteration(self, weights, metrics, step):
        self.history.append({
            'step': step,
            'weights': weights.copy(),
            'metrics': metrics.copy()
        })
        
    def generate_dashboard(self, output_path: str):
        df = pd.DataFrame(self.history)
        fig = go.Figure()
        for reward in df['weights'][0].keys():
            weights = [h['weights'][reward] for h in self.history]
            fig.add_trace(go.Scatter(
                x=df['step'],
                y=weights,
                name=f'{reward} weight'
            ))
        fig.write_html(output_path)