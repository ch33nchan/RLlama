import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import numpy as np
from typing import Dict, List
import os
import sys

# Add RLlama to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="RLlama Dashboard",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_training_data(analysis_file: str = "rllama_training_analysis.json", 
                      stats_file: str = "training_stats.json") -> tuple:
    """Load training analysis and statistics"""
    analysis_data = {}
    training_stats = []
    
    try:
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
    except Exception as e:
        st.warning(f"Could not load analysis file: {e}")
    
    try:
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                training_stats = json.load(f)
    except Exception as e:
        st.warning(f"Could not load training stats: {e}")
    
    return analysis_data, training_stats

def create_component_breakdown_chart(analysis_data: Dict) -> go.Figure:
    """Create component contribution breakdown"""
    avg_rewards = analysis_data.get('avg_rewards', {})
    
    if not avg_rewards:
        return go.Figure().add_annotation(text="No component data available")
    
    components = list(avg_rewards.keys())
    values = list(avg_rewards.values())
    
    fig = px.bar(
        x=components, 
        y=values,
        title="Average Component Contributions",
        labels={'x': 'Reward Components', 'y': 'Average Contribution'},
        color=values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False
    )
    
    return fig

def create_training_progress_chart(training_stats: List[Dict]) -> go.Figure:
    """Create training progress over time"""
    if not training_stats:
        return go.Figure().add_annotation(text="No training data available")
    
    steps = [stat['step'] for stat in training_stats]
    ppo_rewards = [stat['mean_ppo_reward'] for stat in training_stats]
    rllama_rewards = [stat['mean_rllama_reward'] for stat in training_stats]
    kl_divergence = [stat['objective_kl'] for stat in training_stats]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PPO Rewards', 'RLlama Rewards', 'KL Divergence', 'Reward Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # PPO Rewards
    fig.add_trace(
        go.Scatter(x=steps, y=ppo_rewards, name='PPO Rewards', line=dict(color='blue')),
        row=1, col=1
    )
    
    # RLlama Rewards
    fig.add_trace(
        go.Scatter(x=steps, y=rllama_rewards, name='RLlama Rewards', line=dict(color='green')),
        row=1, col=2
    )
    
    # KL Divergence
    fig.add_trace(
        go.Scatter(x=steps, y=kl_divergence, name='KL Divergence', line=dict(color='red')),
        row=2, col=1
    )
    
    # Comparison
    fig.add_trace(
        go.Scatter(x=steps, y=ppo_rewards, name='PPO', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=steps, y=rllama_rewards, name='RLlama', line=dict(color='green')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def create_reward_distribution_chart(analysis_data: Dict) -> go.Figure:
    """Create reward distribution visualization"""
    overall_stats = analysis_data.get('overall_statistics', {})
    
    if not overall_stats:
        return go.Figure().add_annotation(text="No distribution data available")
    
    raw_stats = overall_stats.get('raw_rewards', {})
    norm_stats = overall_stats.get('normalized_rewards', {})
    
    fig = go.Figure()
    
    # Create mock distribution data (in real app, you'd have the actual reward values)
    if raw_stats.get('mean') and raw_stats.get('std'):
        x_raw = np.linspace(
            raw_stats['mean'] - 3*raw_stats['std'], 
            raw_stats['mean'] + 3*raw_stats['std'], 
            100
        )
        y_raw = np.exp(-0.5 * ((x_raw - raw_stats['mean']) / raw_stats['std'])**2)
        
        fig.add_trace(go.Scatter(
            x=x_raw, y=y_raw, 
            name='Raw Rewards',
            fill='tonexty',
            line=dict(color='lightblue')
        ))
    
    if norm_stats.get('mean') is not None and norm_stats.get('std'):
        x_norm = np.linspace(-3, 3, 100)
        y_norm = np.exp(-0.5 * x_norm**2)
        
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            name='Normalized Rewards',
            fill='tonexty',
            line=dict(color='lightgreen')
        ))
    
    fig.update_layout(
        title="Reward Distribution Comparison",
        xaxis_title="Reward Value",
        yaxis_title="Density",
        height=400
    )
    
    return fig

def create_component_trends_chart(analysis_data: Dict) -> go.Figure:
    """Create component trends over time"""
    component_trends = analysis_data.get('component_trends', {})
    
    if not component_trends:
        return go.Figure().add_annotation(text="No trend data available")
    
    fig = go.Figure()
    
    for component, values in component_trends.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            name=component,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Component Trends Over Time",
        xaxis_title="Training Step",
        yaxis_title="Component Contribution",
        height=400
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Title and header
    st.title("🦙 RLlama Training Dashboard")
    st.markdown("**Real-time monitoring of reward engineering and training progress**")
    
    # Sidebar for controls
    st.sidebar.header("Dashboard Controls")
    
    # File upload option
    uploaded_analysis = st.sidebar.file_uploader(
        "Upload Analysis File", 
        type=['json'],
        help="Upload rllama_training_analysis.json"
    )
    
    uploaded_stats = st.sidebar.file_uploader(
        "Upload Training Stats", 
        type=['json'],
        help="Upload training_stats.json"
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 10s)", value=False)
    
    if auto_refresh:
        # Auto-refresh every 10 seconds
        import time
        time.sleep(10)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Load data
    if uploaded_analysis and uploaded_stats:
        analysis_data = json.load(uploaded_analysis)
        training_stats = json.load(uploaded_stats)
    else:
        analysis_data, training_stats = load_training_data()
    
    # Main dashboard layout
    if not analysis_data and not training_stats:
        st.warning("No training data found. Please run a training session or upload data files.")
        st.info("Run: `python examples/trl_rllama_example.py` to generate training data.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_steps = analysis_data.get('total_steps', 0)
        st.metric("Total Training Steps", total_steps)
    
    with col2:
        overall_stats = analysis_data.get('overall_statistics', {})
        raw_mean = overall_stats.get('raw_rewards', {}).get('mean', 0)
        st.metric("Mean Raw Reward", f"{raw_mean:.3f}")
    
    with col3:
        norm_mean = overall_stats.get('normalized_rewards', {}).get('mean', 0)
        st.metric("Mean Normalized Reward", f"{norm_mean:.3f}")
    
    with col4:
        num_components = len(analysis_data.get('avg_rewards', {}))
        st.metric("Active Components", num_components)
    
    # Charts section
    st.header("📊 Training Analysis")
    
    # Component breakdown
    st.subheader("Reward Component Breakdown")
    component_chart = create_component_breakdown_chart(analysis_data)
    st.plotly_chart(component_chart, use_container_width=True)
    
    # Training progress
    if training_stats:
        st.subheader("Training Progress")
        progress_chart = create_training_progress_chart(training_stats)
        st.plotly_chart(progress_chart, use_container_width=True)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Reward Distribution")
        dist_chart = create_reward_distribution_chart(analysis_data)
        st.plotly_chart(dist_chart, use_container_width=True)
    
    with col2:
        st.subheader("Component Trends")
        trends_chart = create_component_trends_chart(analysis_data)
        st.plotly_chart(trends_chart, use_container_width=True)
    
    # Detailed statistics
    st.header("📋 Detailed Statistics")
    
    # Component statistics table
    if 'reward_statistics' in analysis_data:
        st.subheader("Component Statistics")
        stats_data = []
        for comp_name, stats in analysis_data['reward_statistics'].items():
            stats_data.append({
                'Component': comp_name,
                'Mean': f"{stats['mean']:.4f}",
                'Std': f"{stats['std']:.4f}",
                'Min': f"{stats['min']:.4f}",
                'Max': f"{stats['max']:.4f}",
                'Count': stats['count']
            })
        
        df = pd.DataFrame(stats_data)
        st.dataframe(df, use_container_width=True)
    
    # Training examples
    if training_stats:
        st.subheader("Recent Training Examples")
        recent_examples = training_stats[-5:]  # Last 5 examples
        
        for i, example in enumerate(recent_examples):
            with st.expander(f"Step {example['step']} - Reward: {example['example_reward']:.3f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Query:**")
                    st.write(example['example_query'])
                with col2:
                    st.write("**Response:**")
                    st.write(example['example_response'])
                
                st.write(f"**Reward:** {example['example_reward']:.3f}")
    
    # Export section
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Analysis JSON"):
            if analysis_data:
                st.download_button(
                    label="📥 Download Analysis",
                    data=json.dumps(analysis_data, indent=2),
                    file_name="rllama_analysis_export.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("Download Training Stats JSON"):
            if training_stats:
                st.download_button(
                    label="📥 Download Stats",
                    data=json.dumps(training_stats, indent=2),
                    file_name="training_stats_export.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()