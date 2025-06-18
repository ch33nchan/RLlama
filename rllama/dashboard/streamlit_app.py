# rllama/dashboard/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import glob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RLlama components
try:
    from rllama.dashboard import RewardTracker, RewardVisualizer
except ImportError:
    # Fallback for standalone dashboard
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    from rllama.dashboard import RewardTracker, RewardVisualizer

# Get log directory from environment or use default
LOG_DIR = os.environ.get("RLLAMA_LOG_DIR", "./rllama_logs")
REWARD_LOG_DIR = os.environ.get("RLLAMA_REWARD_LOG_DIR", "./rllama_reward_logs")

# Set page config
st.set_page_config(
    page_title="RLlama Dashboard",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for navigation
st.sidebar.title("🦙 RLlama Dashboard")
page = st.sidebar.radio("Select Page", [
    "Overview",
    "Reward Components",
    "Episode Analysis",
    "Run Comparison",
    "Settings"
])

# Function to load all episode data
def load_episode_data(log_dir):
    episode_file = os.path.join(log_dir, "episodes.jsonl")
    episodes = []
    
    if os.path.exists(episode_file):
        with open(episode_file, 'r') as f:
            for line in f:
                try:
                    episode = json.loads(line)
                    episodes.append(episode)
                except:
                    pass
    
    return episodes

# Function to load reward visualizer
@st.cache_resource
def load_reward_visualizer():
    return RewardVisualizer(REWARD_LOG_DIR)

# Page: Overview
if page == "Overview":
    st.title("📊 RLlama Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Overview")
        # Load episode data
        episodes = load_episode_data(LOG_DIR)
        
        if episodes:
            # Basic stats
            total_episodes = len(episodes)
            total_steps = episodes[-1]["total_steps"] if episodes else 0
            avg_reward = np.mean([ep.get("episode_reward", 0) for ep in episodes])
            avg_length = np.mean([ep.get("episode_steps", 0) for ep in episodes])
            
            st.markdown(f"""
            ### Statistics
            - **Total Episodes:** {total_episodes}
            - **Total Steps:** {total_steps}
            - **Avg. Episode Reward:** {avg_reward:.4f}
            - **Avg. Episode Length:** {avg_length:.2f} steps
            """)
            
            # Plot rolling reward
            if len(episodes) > 1:
                rewards = [ep.get("episode_reward", 0) for ep in episodes]
                df = pd.DataFrame({
                    "episode": range(len(rewards)),
                    "reward": rewards
                })
                
                # Add rolling average
                window = min(10, len(rewards))
                df["rolling_reward"] = df["reward"].rolling(window=window).mean()
                
                # Plot
                fig = px.line(df, x="episode", y=["reward", "rolling_reward"], 
                              title=f"Episode Rewards (with {window}-ep rolling avg)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No episode data found. Start training to generate data.")
    
    with col2:
        st.subheader("Recent Episodes")
        # Show recent episodes
        if episodes:
            recent = episodes[-10:]
            recent.reverse()  # Show newest first
            
            # Create a nice table
            data = []
            for ep in recent:
                data.append({
                    "Episode": ep.get("episode", "N/A"),
                    "Steps": ep.get("episode_steps", "N/A"),
                    "Reward": round(ep.get("episode_reward", 0), 4),
                    "Time": ep.get("timestamp", "").split("T")[0]
                })
            
            st.table(pd.DataFrame(data))
        else:
            st.info("No episodes recorded yet.")
    
    # Environment info
    st.subheader("Environment")
    
    # Try to find environment info
    env_file = os.path.join(LOG_DIR, "environment_info.json")
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_info = json.load(f)
        
        st.json(env_info)
    else:
        st.info("No environment information available.")

# Page: Reward Components
elif page == "Reward Components":
    st.title("🧩 Reward Components")
    
    # Load reward visualizer
    visualizer = load_reward_visualizer()
    
    # Get available logs
    available_logs = list(visualizer.trackers.keys())
    
    if not available_logs:
        st.warning("No reward logs found. Start training to generate component data.")
    else:
        # Select log
        selected_log = st.selectbox("Select Log", available_logs)
        
        # Get components
        tracker = visualizer.trackers[selected_log]
        components = list(tracker.reward_history.keys())
        
        # Select components to visualize
        selected_components = st.multiselect("Select Components", components, 
                                           default=["final_reward"] if "final_reward" in components else components[:3])
        
        # Visualization options
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider("Smoothing Window Size", 1, 50, 10)
        with col2:
            show_episodes = st.checkbox("Show Episode Boundaries", value=True)
        
        # Plot the components
        if selected_components:
            fig = visualizer.plot_interactive(selected_log, selected_components, window_size, show_episodes)
            st.plotly_chart(fig, use_container_width=True)
            
            # Component statistics
            st.subheader("Component Statistics")
            
            stats = {}
            for comp in selected_components:
                values = tracker.reward_history.get(comp, [])
                if values:
                    stats[comp] = {
                        "Mean": np.mean(values),
                        "Std Dev": np.std(values),
                        "Min": np.min(values),
                        "Max": np.max(values),
                        "Median": np.median(values)
                    }
            
            st.table(pd.DataFrame(stats))
        else:
            st.info("Select at least one component to visualize.")

# Page: Episode Analysis
elif page == "Episode Analysis":
    st.title("📈 Episode Analysis")
    
    # Load episode data
    episodes = load_episode_data(LOG_DIR)
    
    if not episodes:
        st.warning("No episode data found. Start training to generate data.")
    else:
        # Convert to DataFrame for analysis
        df = pd.DataFrame(episodes)
        
        # Basic episode metrics
        st.subheader("Episode Metrics")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("Total Episodes", len(df))
        
        with metrics_col2:
            if "episode_reward" in df.columns:
                st.metric("Avg Reward", f"{df['episode_reward'].mean():.4f}")
        
        with metrics_col3:
            if "episode_steps" in df.columns:
                st.metric("Avg Steps", f"{df['episode_steps'].mean():.1f}")
        
        # Plot episode length vs reward
        st.subheader("Episode Length vs. Reward")
        
        if "episode_steps" in df.columns and "episode_reward" in df.columns:
            scatter_fig = px.scatter(df, x="episode_steps", y="episode_reward", 
                                   color="episode",
                                   title="Episode Length vs. Reward",
                                   labels={"episode_steps": "Episode Length (steps)",
                                          "episode_reward": "Episode Reward"})
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Episode reward distribution
        st.subheader("Reward Distribution")
        
        if "episode_reward" in df.columns:
            hist_fig = px.histogram(df, x="episode_reward", nbins=20,
                                  title="Distribution of Episode Rewards")
            st.plotly_chart(hist_fig, use_container_width=True)
        
        # Detailed episode list
        st.subheader("Episode Details")
        
        # Allow filtering of episodes
        min_reward = float(df["episode_reward"].min()) if "episode_reward" in df.columns else 0
        max_reward = float(df["episode_reward"].max()) if "episode_reward" in df.columns else 0
        
        reward_range = st.slider("Filter by Reward Range", min_reward, max_reward, 
                               (min_reward, max_reward))
        
        # Filter episodes
        filtered_df = df
        if "episode_reward" in df.columns:
            filtered_df = df[(df["episode_reward"] >= reward_range[0]) & 
                           (df["episode_reward"] <= reward_range[1])]
        
        # Show the table
        st.dataframe(filtered_df)

# Page: Run Comparison
elif page == "Run Comparison":
    st.title("🔍 Run Comparison")
    
    # Load reward visualizer
    visualizer = load_reward_visualizer()
    
    # Get available logs
    available_logs = list(visualizer.trackers.keys())
    
    if len(available_logs) < 2:
        st.warning("Need at least two runs to compare. Only found {len(available_logs)} runs.")
    else:
        # Select logs to compare
        selected_logs = st.multiselect("Select Runs to Compare", available_logs,
                                     default=available_logs[:2])
        
        if len(selected_logs) < 2:
            st.info("Select at least two runs to compare.")
        else:
            # Get common components
            common_components = set()
            for log in selected_logs:
                tracker = visualizer.trackers[log]
                if not common_components:
                    common_components = set(tracker.reward_history.keys())
                else:
                    common_components = common_components.intersection(set(tracker.reward_history.keys()))
            
            # Select component to compare
            selected_component = st.selectbox("Select Component to Compare", sorted(list(common_components)),
                                           index=0 if "final_reward" in common_components else 0)
            
            # Visualization options
            window_size = st.slider("Smoothing Window Size", 1, 50, 10)
            
            # Plot comparison
            fig = visualizer.plot_component_comparison(selected_logs, selected_component, window_size)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics comparison
            st.subheader("Statistical Comparison")
            
            stats = {}
            for log in selected_logs:
                tracker = visualizer.trackers[log]
                values = tracker.reward_history.get(selected_component, [])
                if values:
                    stats[log] = {
                        "Mean": np.mean(values),
                        "Std Dev": np.std(values),
                        "Min": np.min(values),
                        "Max": np.max(values),
                        "Median": np.median(values)
                    }
            
            st.table(pd.DataFrame(stats))

# Page: Settings
elif page == "Settings":
    st.title("⚙️ Settings")
    
    st.header("Log Directories")
    
    # Display current log directories
    st.markdown(f"""
    * **Episode Logs:** `{LOG_DIR}`
    * **Reward Logs:** `{REWARD_LOG_DIR}`
    """)
    
    # Option to change log directories
    new_log_dir = st.text_input("New Episode Log Directory", LOG_DIR)
    new_reward_log_dir = st.text_input("New Reward Log Directory", REWARD_LOG_DIR)
    
    if st.button("Update Directories"):
        # This won't persist across app restarts without additional storage
        os.environ["RLLAMA_LOG_DIR"] = new_log_dir
        os.environ["RLLAMA_REWARD_LOG_DIR"] = new_reward_log_dir
        st.success("Directories updated! Please refresh the page.")
    
    st.header("About")
    
    st.markdown("""
    ## RLlama Dashboard
    
    This dashboard provides visualization and analysis tools for reinforcement learning
    experiments conducted with the RLlama library.
    
    ### Features
    
    * Track episode rewards and statistics
    * Visualize reward components
    * Compare different runs
    * Analyze episode performance
    
    ### Documentation
    
    For more information, see the [RLlama documentation](https://github.com/ch33nchan/RLlama).
    """)

# Footer
st.markdown("---")
st.markdown("RLlama Dashboard | v0.3.0")