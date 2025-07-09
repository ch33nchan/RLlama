import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import time

# Set page config
st.set_page_config(
    page_title="RLlama - RL Laboratory",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions for TE-styled components
def te_knob(label, key, min_val, max_val, default_val, step=0.0001):
    """Create a more interactive knob control with immediate feedback"""
    value = st.slider(
        label, 
        min_value=min_val, 
        max_value=max_val, 
        value=default_val,
        step=step,
        format="%.5f" if step < 0.001 else "%.3f",
        key=key
    )
    
    # Add visual knob representation
    st.markdown(f"""
    <div style="margin-top:5px; margin-bottom:15px; width:100%; height:4px; background:#333; position:relative;">
        <div style="position:absolute; left:{((value-min_val)/(max_val-min_val))*100}%; top:-6px; width:16px; height:16px; background:#FF9900; border-radius:50%; transform:translateX(-50%);"></div>
    </div>
    """, unsafe_allow_html=True)
    
    return value

def te_toggle(label, key, default=False):
    """Improved toggle with better visual feedback"""
    col1, col2 = st.columns([1, 6])
    
    with col1:
        value = st.checkbox("", value=default, key=key, label_visibility="collapsed")
    
    # Display styled toggle
    active_class = "active" if value else ""
    with col2:
        st.markdown(f"""
        <div style="margin-bottom:15px;">
            <div class="te-toggle {active_class}" style="display:inline-block; width:50px; height:24px; background-color:{('#FF9900' if value else '#333333')}; border-radius:12px; position:relative;">
                <div style="position:absolute; width:20px; height:20px; border-radius:10px; background-color:#FFFFFF; top:2px; {'right:2px' if value else 'left:2px'}; transition:all 0.3s;"></div>
            </div>
            <span style="margin-left:10px; font-family:'Space Mono', monospace; color:#AAAAAA; font-size:14px;">{label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    return value

# Sidebar navigation
def sidebar():
    st.sidebar.image("https://i.imgur.com/XQMapI6.png", width=100)  # Placeholder for RLlama logo
    st.sidebar.title("RLlama")
    st.sidebar.caption("Reinforcement Learning Laboratory")
    
    # Page selection
    page = st.sidebar.radio(
        "NAVIGATION",
        ["ALGORITHMS", "ENVIRONMENTS", "TRAINING LAB", "VISUALIZATION"],
        key="page_selection"
    )
    
    # RLlama toggle - more prominent in sidebar
    st.sidebar.markdown("### CONFIGURATION")
    use_rllama = te_toggle("USE RLLAMA", "use_rllama", True)
    
    # App version info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Version: 1.0.0")
    st.sidebar.caption(f"System Time: 2025-07-09 21:15:14")
    st.sidebar.caption(f"User: ch33nchan")
    
    return page, use_rllama

# ALGORITHMS PAGE
def show_algorithms_page(use_rllama):
    st.markdown("<h2>ALGORITHMS</h2>", unsafe_allow_html=True)
    
    # Algorithm selection
    algorithm = st.selectbox(
        "SELECT ALGORITHM",
        ["SAC (Soft Actor-Critic)", "PPO (Proximal Policy Optimization)", 
         "DQN (Deep Q-Network)", "DDPG (Deep Deterministic Policy Gradient)"],
        key="algorithm"
    )
    
    # Algorithm content
    if algorithm == "SAC (Soft Actor-Critic)":
        # SAC Algorithm overview
        st.markdown("""
        <div class="te-module" data-title="ALGORITHM">
            <h3>Soft Actor-Critic (SAC)</h3>
            <p>SAC is an off-policy actor-critic algorithm that:</p>
            <ul>
                <li>Maximizes both expected return and entropy</li>
                <li>Uses two Q-functions to mitigate positive bias</li>
                <li>Automatically tunes temperature parameter</li>
                <li>Works well for continuous action spaces</li>
            </ul>
            <div style="margin-top:15px;">
                <span class="te-tag">OFF-POLICY</span>
                <span class="te-tag">CONTINUOUS</span>
                <span class="te-tag">SAMPLE-EFFICIENT</span>
                <span class="te-tag">MAXIMUM ENTROPY</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive parameter controls with TE-inspired components
        st.markdown("<h3>ALGORITHM PARAMETERS</h3>", unsafe_allow_html=True)
        
        # Create parameter tuning panels with better spacing
        with st.container():
            st.markdown('<div class="te-control-panel">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("<div style='padding:10px; background:#222; border-radius:4px; margin-bottom:15px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#FF9900; font-family:Space Mono; font-size:14px; margin-bottom:15px;'>LEARNING RATES</h4>", unsafe_allow_html=True)
                actor_lr = te_knob("ACTOR LR", "actor_lr", 0.0001, 0.001, 0.0003, 0.00001)
                critic_lr = te_knob("CRITIC LR", "critic_lr", 0.0001, 0.001, 0.0003, 0.00001)
                alpha_lr = te_knob("ALPHA LR", "alpha_lr", 0.0001, 0.001, 0.0003, 0.00001)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div style='padding:10px; background:#222; border-radius:4px; margin-bottom:15px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#FF9900; font-family:Space Mono; font-size:14px; margin-bottom:15px;'>NETWORK SETTINGS</h4>", unsafe_allow_html=True)
                batch_size = st.select_slider(
                    "BATCH SIZE",
                    options=[32, 64, 128, 256, 512],
                    value=256,
                    key="batch_size"
                )
                buffer_size = st.select_slider(
                    "BUFFER SIZE",
                    options=[10000, 50000, 100000, 500000, 1000000],
                    value=100000,
                    key="buffer_size"
                )
                hidden_size = st.select_slider(
                    "HIDDEN SIZE",
                    options=[64, 128, 256, 512],
                    value=256,
                    key="hidden_size"
                )
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("<div style='padding:10px; background:#222; border-radius:4px; margin-bottom:15px;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color:#FF9900; font-family:Space Mono; font-size:14px; margin-bottom:15px;'>ALGORITHM SETTINGS</h4>", unsafe_allow_html=True)
                gamma = st.slider("DISCOUNT (γ)", min_value=0.9, max_value=0.999, value=0.99, format="%.3f", key="gamma")
                tau = st.slider("TARGET UPDATE (τ)", min_value=0.001, max_value=0.01, value=0.005, format="%.4f", key="tau")
                auto_entropy = te_toggle("AUTO ENTROPY", "auto_entropy", True)
                initial_alpha = st.slider("INITIAL ALPHA", min_value=0.1, max_value=1.0, value=0.2, format="%.2f", key="init_alpha", disabled=auto_entropy)
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance visualization
        st.markdown("<h3>ALGORITHM PERFORMANCE</h3>", unsafe_allow_html=True)
        
        # Environment selection and training parameters
        col1, col2 = st.columns([1, 3])
        
        with col1:
            env_name = st.selectbox(
                "ENVIRONMENT",
                ["Pendulum-v1", "HalfCheetah-v4", "Humanoid-v4"],
                key="env_name"
            )
            
            training_steps = st.slider("TRAINING STEPS", min_value=5000, max_value=100000, value=20000, step=5000, key="training_steps")
            
            # Create better styled run button
            st.markdown("""
            <div style="margin-top:20px; margin-bottom:20px;">
                <button id="run_sim_btn" type="button" style="background-color:#FF9900; color:#000; border:none; padding:10px 20px; border-radius:4px; font-family:'Space Mono', monospace; cursor:pointer; width:100%;">
                    ▶ RUN SIMULATION
                </button>
            </div>
            """, unsafe_allow_html=True)
            run_button = st.button("Run Simulation", key="run_simulation", label_visibility="collapsed")
        
        # Comparison mode selection - show both with and without RLlama
        with col2:
            st.markdown("""
            <div style="background:#1A1A1A; padding:10px; border-radius:4px; margin-bottom:15px;">
                <h4 style="color:#FF9900; font-family:Space Mono; font-size:14px; margin-bottom:10px;">COMPARISON MODE</h4>
            """, unsafe_allow_html=True)
            
            show_comparison = st.checkbox("Show comparison with/without RLlama", value=True, key="show_comparison")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        if run_button:
            with st.spinner("Simulating..."):
                # Create a progress bar with TE styling
                progress_placeholder = st.empty()
                
                for i in range(11):
                    progress_html = f"""
                    <div style="width: 100%; background-color: #1A1A1A; height: 24px; border-radius: 2px; overflow: hidden; margin: 1em 0; position: relative;">
                        <div style="width: {i*10}%; background-color: #FF9900; height: 100%;"></div>
                        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-family: 'Space Mono', monospace; font-size: 0.8em;">
                            {i*10}%
                        </div>
                    </div>
                    """
                    progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
                    time.sleep(0.1)
                
                # Generate synthetic data
                np.random.seed(42)
                steps = np.linspace(0, training_steps, 100)
                
                # Calculate parameters influence factors
                lr_factor = 0.5 + np.tanh((np.log10(actor_lr) + 4) * 2) * 0.5  # Optimal around 3e-4
                batch_factor = 0.5 + np.tanh((batch_size - 128) / 128) * 0.5  # Optimal around 256
                gamma_factor = 0.5 + np.tanh((gamma - 0.97) * 30) * 0.5  # Optimal around 0.99
                tau_factor = 0.5 + np.tanh((np.log10(tau) + 3) * 2) * 0.5  # Optimal around 0.005
                entropy_factor = 1.2 if auto_entropy else (0.8 * (1 - abs(initial_alpha - 0.2)))
                
                # Different baseline for different environments
                if env_name == "Pendulum-v1":
                    baseline = -1600 + 1500 * (1 - np.exp(-steps/5000))
                    reward_scale = -1600
                    max_reward = -100
                elif env_name == "HalfCheetah-v4":
                    baseline = 500 * (1 - np.exp(-steps/10000))
                    reward_scale = 0
                    max_reward = 500
                else:  # Humanoid-v4
                    baseline = 1000 * (1 - np.exp(-steps/20000))
                    reward_scale = 0
                    max_reward = 1000
                
                # Calculate performance with and without RLlama
                performance_with_rllama = baseline * lr_factor * batch_factor * gamma_factor * tau_factor * entropy_factor
                performance_without_rllama = baseline * 0.6  # Simulating lower performance without RLlama
                
                # Add noise and variations
                noise_with = np.random.normal(0, abs(max_reward - reward_scale) * 0.05, size=len(steps))
                noise_without = np.random.normal(0, abs(max_reward - reward_scale) * 0.08, size=len(steps))
                variations = np.sin(steps / 1000) * abs(max_reward - reward_scale) * 0.02
                
                performance_with_rllama += noise_with + variations
                performance_without_rllama += noise_without + variations * 0.5
                
                # Create DataFrame for chart
                if show_comparison:
                    # Show both lines for comparison
                    data = pd.DataFrame({
                        'Step': np.tile(steps, 2),
                        'Reward': np.concatenate([performance_with_rllama, performance_without_rllama]),
                        'Method': np.repeat(['RLlama', 'Without RLlama'], len(steps))
                    })
                    
                    chart = alt.Chart(data).mark_line(strokeWidth=2).encode(
                        x=alt.X('Step', title='TRAINING STEPS'),
                        y=alt.Y('Reward', title='AVERAGE REWARD'),
                        color=alt.Color('Method', scale=alt.Scale(
                            domain=['RLlama', 'Without RLlama'],
                            range=['#FF9900', '#00CCFF']
                        )),
                        tooltip=['Method', 'Step', 'Reward']
                    ).properties(
                        width=700,
                        height=400,
                        title=f"SAC Performance Comparison: {env_name}"
                    ).configure_axis(
                        labelColor='#CCCCCC',
                        titleColor='#FFFFFF',
                        grid=True,
                        gridColor='#333333'
                    ).configure_title(
                        color='#FFFFFF',
                        fontSize=16,
                        font='Space Mono'
                    ).configure_legend(
                        labelColor='#CCCCCC',
                        titleColor='#FFFFFF',
                        orient='top'
                    )
                else:
                    # Show single line based on the use_rllama toggle
                    display_data = pd.DataFrame({
                        'Step': steps,
                        'Reward': performance_with_rllama if use_rllama else performance_without_rllama
                    })
                    
                    line_color = '#FF9900' if use_rllama else '#00CCFF'
                    chart_title = f"SAC Performance with{'out' if not use_rllama else ''} RLlama: {env_name}"
                    
                    chart = alt.Chart(display_data).mark_line(color=line_color, strokeWidth=2).encode(
                        x=alt.X('Step', title='TRAINING STEPS'),
                        y=alt.Y('Reward', title='AVERAGE REWARD'),
                        tooltip=['Step', 'Reward']
                    ).properties(
                        width=700,
                        height=400,
                        title=chart_title
                    ).configure_axis(
                        labelColor='#CCCCCC',
                        titleColor='#FFFFFF',
                        grid=True,
                        gridColor='#333333'
                    ).configure_title(
                        color='#FFFFFF',
                        fontSize=16,
                        font='Space Mono'
                    )
                
                # Display chart with TE styling
                st.markdown('<div class="te-chart-container">', unsafe_allow_html=True)
                st.altair_chart(chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display metrics
                st.markdown("<h4>PERFORMANCE METRICS</h4>", unsafe_allow_html=True)
                
                final_reward_with = performance_with_rllama[-1]
                final_reward_without = performance_without_rllama[-1]
                
                if show_comparison or use_rllama:
                    display_reward = final_reward_with
                    comparison_reward = final_reward_without
                else:
                    display_reward = final_reward_without
                    comparison_reward = final_reward_with
                
                comparison_pct = ((display_reward - comparison_reward) / abs(comparison_reward)) * 100 if abs(comparison_reward) > 0 else 0
                
                # More attractive metrics display
                st.markdown('<div style="display:flex; gap:20px; margin-bottom:20px;">', unsafe_allow_html=True)
                
                # Metric 1: Final Reward
                st.markdown(f"""
                <div class="te-metric" style="flex:1; background:#222; padding:20px; border-radius:4px; text-align:center;">
                    <div style="font-family:'Space Mono', monospace; font-size:28px; color:#FF9900; margin-bottom:10px;">{display_reward:.1f}</div>
                    <div style="font-family:'Space Mono', monospace; font-size:14px; color:#AAAAAA;">FINAL REWARD</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metric 2: Training Efficiency
                efficiency = (lr_factor * batch_factor * gamma_factor * tau_factor * entropy_factor) * 100 if use_rllama or show_comparison else 60
                st.markdown(f"""
                <div class="te-metric" style="flex:1; background:#222; padding:20px; border-radius:4px; text-align:center;">
                    <div style="font-family:'Space Mono', monospace; font-size:28px; color:#FF9900; margin-bottom:10px;">{efficiency:.0f}%</div>
                    <div style="font-family:'Space Mono', monospace; font-size:14px; color:#AAAAAA;">TRAINING EFFICIENCY</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metric 3: Comparison
                sign = "+" if comparison_pct > 0 else ""
                st.markdown(f"""
                <div class="te-metric" style="flex:1; background:#222; padding:20px; border-radius:4px; text-align:center;">
                    <div style="font-family:'Space Mono', monospace; font-size:28px; color:#{'00FF00' if comparison_pct > 0 else 'FF0000'}; margin-bottom:10px;">{sign}{comparison_pct:.0f}%</div>
                    <div style="font-family:'Space Mono', monospace; font-size:14px; color:#AAAAAA;">VS {'WITHOUT' if use_rllama or show_comparison else 'WITH'} RLLAMA</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Parameter impact analysis
                st.markdown("<h4>PARAMETER IMPACT ANALYSIS</h4>", unsafe_allow_html=True)
                
                # Create visually appealing parameter impact display
                st.markdown("""
                <div style="background:#1A1A1A; padding:15px; border-radius:4px; margin-bottom:20px;">
                    <table style="width:100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #333; text-align:left;">
                            <th style="padding: 10px; color: #FF9900; font-family: 'Space Mono', monospace; width:20%;">PARAMETER</th>
                            <th style="padding: 10px; color: #FF9900; font-family: 'Space Mono', monospace; width:20%;">YOUR VALUE</th>
                            <th style="padding: 10px; color: #FF9900; font-family: 'Space Mono', monospace; width:35%;">IMPACT</th>
                            <th style="padding: 10px; color: #FF9900; font-family: 'Space Mono', monospace; width:25%;">RECOMMENDATION</th>
                        </tr>
                """, unsafe_allow_html=True)
                
                # Actor Learning Rate
                impact_color_actor = "#00FF00" if lr_factor > 0.8 else "#FFCC00" if lr_factor > 0.5 else "#FF3300"
                rec_actor = "Optimal" if lr_factor > 0.8 else f"Try {'increasing' if actor_lr < 0.0003 else 'decreasing'} to 0.0003"
                st.markdown(f"""
                <tr style="border-bottom: 1px solid #222;">
                    <td style="padding: 10px; color: #DDD;">Actor Learning Rate</td>
                    <td style="padding: 10px; color: #DDD;">{actor_lr:.5f}</td>
                    <td style="padding: 10px;">
                        <div style="width:100%; height:20px; background:#333; border-radius:2px; overflow:hidden;">
                            <div style="width:{lr_factor*100}%; height:100%; background:{impact_color_actor};"></div>
                        </div>
                    </td>
                    <td style="padding: 10px; color: {impact_color_actor};">{rec_actor}</td>
                </tr>
                """, unsafe_allow_html=True)
                
                # Batch Size
                impact_color_batch = "#00FF00" if batch_factor > 0.8 else "#FFCC00" if batch_factor > 0.5 else "#FF3300"
                rec_batch = "Optimal" if batch_factor > 0.8 else f"Try increasing to 256"
                st.markdown(f"""
                <tr style="border-bottom: 1px solid #222;">
                    <td style="padding: 10px; color: #DDD;">Batch Size</td>
                    <td style="padding: 10px; color: #DDD;">{batch_size}</td>
                    <td style="padding: 10px;">
                        <div style="width:100%; height:20px; background:#333; border-radius:2px; overflow:hidden;">
                            <div style="width:{batch_factor*100}%; height:100%; background:{impact_color_batch};"></div>
                        </div>
                    </td>
                    <td style="padding: 10px; color: {impact_color_batch};">{rec_batch}</td>
                </tr>
                """, unsafe_allow_html=True)
                
                # Discount Factor (gamma)
                impact_color_gamma = "#00FF00" if gamma_factor > 0.8 else "#FFCC00" if gamma_factor > 0.5 else "#FF3300"
                rec_gamma = "Optimal" if gamma_factor > 0.8 else f"Try {'increasing' if gamma < 0.99 else 'decreasing'} to 0.99"
                st.markdown(f"""
                <tr style="border-bottom: 1px solid #222;">
                    <td style="padding: 10px; color: #DDD;">Discount (γ)</td>
                    <td style="padding: 10px; color: #DDD;">{gamma:.3f}</td>
                    <td style="padding: 10px;">
                        <div style="width:100%; height:20px; background:#333; border-radius:2px; overflow:hidden;">
                            <div style="width:{gamma_factor*100}%; height:100%; background:{impact_color_gamma};"></div>
                        </div>
                    </td>
                    <td style="padding: 10px; color: {impact_color_gamma};">{rec_gamma}</td>
                </tr>
                """, unsafe_allow_html=True)
                
                # Target Update Rate (tau)
                impact_color_tau = "#00FF00" if tau_factor > 0.8 else "#FFCC00" if tau_factor > 0.5 else "#FF3300"
                rec_tau = "Optimal" if tau_factor > 0.8 else f"Try {'increasing' if tau < 0.005 else 'decreasing'} to 0.005"
                st.markdown(f"""
                <tr style="border-bottom: 1px solid #222;">
                    <td style="padding: 10px; color: #DDD;">Target Update (τ)</td>
                    <td style="padding: 10px; color: #DDD;">{tau:.4f}</td>
                    <td style="padding: 10px;">
                        <div style="width:100%; height:20px; background:#333; border-radius:2px; overflow:hidden;">
                            <div style="width:{tau_factor*100}%; height:100%; background:{impact_color_tau};"></div>
                        </div>
                    </td>
                    <td style="padding: 10px; color: {impact_color_tau};">{rec_tau}</td>
                </tr>
                """, unsafe_allow_html=True)
                
                # Auto Entropy
                impact_color_entropy = "#00FF00" if auto_entropy else "#FF3300"
                rec_entropy = "Optimal" if auto_entropy else "Enable auto entropy tuning"
                st.markdown(f"""
                <tr>
                    <td style="padding: 10px; color: #DDD;">Auto Entropy</td>
                    <td style="padding: 10px; color: #DDD;">{auto_entropy}</td>
                    <td style="padding: 10px;">
                        <div style="width:100%; height:20px; background:#333; border-radius:2px; overflow:hidden;">
                            <div style="width:{entropy_factor/1.2*100}%; height:100%; background:{impact_color_entropy};"></div>
                        </div>
                    </td>
                    <td style="padding: 10px; color: {impact_color_entropy};">{rec_entropy}</td>
                </tr>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
                # Terminal output for training simulation
                st.markdown("<h4>TRAINING LOG</h4>", unsafe_allow_html=True)
                
                # Current time for logs
                current_time = "2025-07-09 21:15:14"
                
                log_lines = []
                interval = training_steps // 10
                
                for i in range(11):
                    step = i * interval
                    if i == 0:
                        log_lines.append(f"[{current_time}] Starting SAC training on {env_name}")
                        log_lines.append(f"[{current_time}] Actor LR: {actor_lr}, Critic LR: {critic_lr}, Alpha LR: {alpha_lr}")
                        log_lines.append(f"[{current_time}] Batch size: {batch_size}, Gamma: {gamma}, Tau: {tau}")
                        log_lines.append(f"[{current_time}] Auto entropy tuning: {auto_entropy}")
                        if not auto_entropy:
                            log_lines.append(f"[{current_time}] Initial alpha value: {initial_alpha}")
                        log_lines.append(f"[{current_time}] Hidden size: {hidden_size}")
                        log_lines.append(f"[{current_time}] Total steps: {training_steps}")
                        log_lines.append(f"[{current_time}] --- Training start ---")
                    else:
                        reward = performance_with_rllama[i*10 - 1] if use_rllama or show_comparison else performance_without_rllama[i*10 - 1]
                        log_lines.append(f"[{current_time}] Step {step}: reward = {reward:.2f}")
                        
                        # Add some interesting events
                        if i == 3:
                            log_lines.append(f"[{current_time}] [INFO] Initial policy stabilization")
                        if i == 5:
                            if auto_entropy:
                                log_lines.append(f"[{current_time}] [INFO] Temperature parameter converging to {0.05:.4f}")
                            else:
                                log_lines.append(f"[{current_time}] [INFO] Fixed temperature parameter may limit exploration")
                        if i == 8:
                            log_lines.append(f"[{current_time}] [INFO] Policy approaching optimal performance")
                    
                # Add final log
                log_lines.append(f"[{current_time}] --- Training complete ---")
                log_lines.append(f"[{current_time}] Final average reward: {display_reward:.2f}")
                
                # Create styled terminal output
                st.markdown(f"""
                <div class="te-terminal" style="background:#000; color:#0F0; font-family:'Space Mono', monospace; font-size:12px; padding:15px; border-radius:4px; max-height:300px; overflow-y:auto; border:1px solid #333;">
                {'<br>'.join(log_lines)}
                </div>
                """, unsafe_allow_html=True)
        
        # Code implementation example
        st.markdown("<h3>IMPLEMENTATION</h3>", unsafe_allow_html=True)
        
        if use_rllama:
            st.markdown("""
            <div class="te-module" data-title="CODE">
            <pre style="background:#000; color:#DDD; padding:15px; border-radius:4px; overflow-x:auto; font-family:'Space Mono', monospace; font-size:12px;">
import rllama
import gymnasium as gym

# Create environment
env = gym.make("Pendulum-v1")

# Create SAC agent with custom parameters
agent = rllama.make_agent(
    "SAC",
    env=env,
    actor_lr=3e-4,
    critic_lr=3e-4,
    alpha_lr=3e-4,
    hidden_sizes=[256, 256],
    batch_size=256,
    buffer_size=100000,
    gamma=0.99,
    tau=0.005,
    use_automatic_entropy_tuning=True
)

# Create and run experiment
config = {"total_timesteps": 20000, "eval_episodes": 5}
experiment = rllama.Experiment(
    name="SAC-Pendulum",
    agent=agent,
    env=env,
    config=config
)
experiment.train(total_steps=20000)
            </pre>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="te-module" data-title="CODE WITHOUT RLLAMA">
            <pre style="background:#000; color:#DDD; padding:15px; border-radius:4px; overflow-x:auto; font-family:'Space Mono', monospace; font-size:12px;">
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import random
from collections import deque

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std
    
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        
        x2 = F.relu(self.fc3(x))
        x2 = F.relu(self.fc4(x2))
        q2 = self.q2(x2)
        
        return q1, q2

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, action_space, hidden_dim=256,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=100000, batch_size=256,
                 use_automatic_entropy_tuning=True):
        
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy parameters from critic to target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            mu, _ = self.actor(state)
            return torch.tanh(mu).cpu().detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]
    
    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state_batch)
            next_q1, next_q2 = self.critic_target(next_state_batch, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Update critic
        current_q1, current_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action, log_prob = self.actor.sample(state_batch)
        q1, q2 = self.critic(state_batch, action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (if automatic entropy tuning is enabled)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

# Training loop
def train_sac(env_name, total_steps=20000):
    env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SACAgent(state_dim, action_dim, env.action_space,
                    hidden_dim=256, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                    gamma=0.99, tau=0.005, buffer_size=100000, batch_size=256,
                    use_automatic_entropy_tuning=True)
    
    episode_rewards = []
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    
    for t in range(total_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if t >= agent.batch_size:
            agent.update_parameters()
        
        if done:
            print(f"Episode {episode_num}, Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            episode_rewards.append(episode_reward)
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_num += 1
    
    env.close()
    return episode_rewards

# Run training
rewards = train_sac("Pendulum-v1", total_steps=20000)
            </pre>
            </div>
            """, unsafe_allow_html=True)
    
    elif algorithm == "PPO (Proximal Policy Optimization)":
        st.write("PPO algorithm details will go here")
    
    elif algorithm == "DQN (Deep Q-Network)":
        st.write("DQN algorithm details will go here")
    
    elif algorithm == "DDPG (Deep Deterministic Policy Gradient)":
        st.write("DDPG algorithm details will go here")

# ENVIRONMENTS PAGE
def show_environments_page():
    st.markdown("<h2>ENVIRONMENTS</h2>", unsafe_allow_html=True)
    
    # TE-inspired environment selector
    env_category = st.selectbox(
        "ENVIRONMENT CATEGORY",
        ["Classic Control", "Box2D", "MuJoCo", "Atari", "Custom"],
        key="env_category"
    )
    
    if env_category == "Classic Control":
        st.markdown("""
        <div class="te-module" data-title="CLASSIC CONTROL">
            <p>Classic control environments are simpler environments that are good for testing algorithms and learning reinforcement learning basics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="te-module" data-title="CARTPOLE-V1">
                <p>A pole is attached to a cart moving along a frictionless track. The goal is to prevent the pole from falling over by applying forces to the cart.</p>
                <div style="margin-top:15px;">
                    <table style="width:100%;">
                        <tr>
                            <td style="padding:4px; color:#AAA;">Action Space:</td>
                            <td style="padding:4px; color:#FFF;">Discrete(2)</td>
                        </tr>
                        <tr>
                            <td style="padding:4px; color:#AAA;">Observation Space:</td>
                            <td style="padding:4px; color:#FFF;">Box(4)</td>
                        </tr>
                        <tr>
                            <td style="padding:4px; color:#AAA;">Reward:</td>
                            <td style="padding:4px; color:#FFF;">+1 for each step</td>
                        </tr>
                        <tr>
                            <td style="padding:4px; color:#AAA;">Recommended Algorithms:</td>
                            <td style="padding:4px; color:#FFF;">DQN, PPO</td>
                        </tr>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="te-module" data-title="PENDULUM-V1">
                <p>A pendulum starts in a random position and the goal is to swing it up and keep it upright.</p>
                <div style="margin-top:15px;">
                    <table style="width:100%;">
                        <tr>
                            <td style="padding:4px; color:#AAA;">Action Space:</td>
                            <td style="padding:4px; color:#FFF;">Box(1)</td>
                        </tr>
                        <tr>
                            <td style="padding:4px; color:#AAA;">Observation Space:</td>
                            <td style="padding:4px; color:#FFF;">Box(3)</td>
                        </tr>
                        <tr>
                            <td style="padding:4px; color:#AAA;">Reward:</td>
                            <td style="padding:4px; color:#FFF;">Based on angle and velocity</td>
                        </tr>
                        <tr>
                            <td style="padding:4px; color:#AAA;">Recommended Algorithms:</td>
                            <td style="padding:4px; color:#FFF;">SAC, DDPG</td>
                        </tr>
                    </table>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif env_category == "Box2D":
        st.write("Box2D environments will go here")
    
    elif env_category == "MuJoCo":
        st.write("MuJoCo environments will go here")
    
    elif env_category == "Atari":
        st.write("Atari environments will go here")
    
    elif env_category == "Custom":
        st.write("Custom environments will go here")

# TRAINING LAB PAGE
def show_training_lab_page():
    st.markdown("<h2>TRAINING LAB</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="te-module" data-title="LAB INFO">
        <p>Welcome to the RLlama Training Lab. This interactive space allows you to configure, train, and evaluate 
        reinforcement learning agents in real-time. Adjust parameters, select environments, and watch the training 
        process unfold.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>ENVIRONMENT SELECTION</h3>", unsafe_allow_html=True)
        
        env_category = st.selectbox(
            "ENVIRONMENT CATEGORY",
            ["Classic Control", "Box2D", "MuJoCo", "Atari"],
            key="lab_env_category"
        )
        
        # Display environments based on category
        if env_category == "Classic Control":
            env_name = st.selectbox(
                "ENVIRONMENT",
                ["CartPole-v1", "Pendulum-v1", "MountainCar-v0", "Acrobot-v1"],
                key="lab_env_name"
            )
        elif env_category == "Box2D":
            env_name = st.selectbox(
                "ENVIRONMENT",
                ["LunarLander-v2", "BipedalWalker-v3", "CarRacing-v2"],
                key="lab_env_name"
            )
        elif env_category == "MuJoCo":
            env_name = st.selectbox(
                "ENVIRONMENT",
                ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4"],
                key="lab_env_name"
            )
        else:  # Atari
            env_name = st.selectbox(
                "ENVIRONMENT",
                ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5", "Enduro-v5"],
                key="lab_env_name"
            )
    
    with col2:
        st.markdown("<h3>ALGORITHM SELECTION</h3>", unsafe_allow_html=True)
        
        # Choose an algorithm
        algorithm = st.selectbox(
            "ALGORITHM",
            ["SAC", "PPO", "DQN", "DDPG"],
            key="lab_algorithm"
        )
        
        # Training parameters
        training_steps = st.slider(
            "TRAINING STEPS",
            min_value=10000,
            max_value=1000000,
            value=100000,
            step=10000,
            key="lab_steps"
        )
        
        eval_episodes = st.slider(
            "EVALUATION EPISODES",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="lab_eval_episodes"
        )
    
    # Parameter adjustment panel based on selected algorithm
    st.markdown("<h3>ALGORITHM PARAMETERS</h3>", unsafe_allow_html=True)
    
    if algorithm == "SAC":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            actor_lr = te_knob("ACTOR LR", "train_actor_lr", 0.0001, 0.001, 0.0003, 0.00001)
            batch_size = st.select_slider(
                "BATCH SIZE",
                options=[32, 64, 128, 256, 512],
                value=256,
                key="train_batch"
            )
        
        with col2:
            critic_lr = te_knob("CRITIC LR", "train_critic_lr", 0.0001, 0.001, 0.0003, 0.00001)
            tau = st.slider("TARGET UPDATE (τ)", min_value=0.001, max_value=0.01, value=0.005, format="%.4f", key="train_tau")
        
        with col3:
            gamma = st.slider("DISCOUNT (γ)", min_value=0.9, max_value=0.999, value=0.99, format="%.3f", key="train_gamma")
            auto_entropy = te_toggle("AUTO ENTROPY", "train_entropy", True)
    
    elif algorithm == "PPO":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = te_knob("LR", "train_ppo_lr", 0.0001, 0.001, 0.0003, 0.00001)
            n_steps = st.select_slider(
                "STEPS PER UPDATE",
                options=[128, 256, 512, 1024, 2048],
                value=2048,
                key="train_ppo_steps"
            )
        
        with col2:
            clip_range = st.slider("CLIP RANGE", min_value=0.1, max_value=0.3, value=0.2, format="%.2f", key="train_ppo_clip")
            n_epochs = st.slider("EPOCHS PER UPDATE", min_value=1, max_value=20, value=10, step=1, key="train_ppo_epochs")
        
        with col3:
            gamma = st.slider("DISCOUNT (γ)", min_value=0.9, max_value=0.999, value=0.99, format="%.3f", key="train_ppo_gamma")
            gae_lambda = st.slider("GAE LAMBDA", min_value=0.9, max_value=1.0, value=0.95, format="%.2f", key="train_ppo_gae")
    
    # Training button - better styled
    st.markdown("""
    <div style="margin-top:30px; margin-bottom:30px; text-align:center;">
        <button type="button" style="background-color:#FF9900; color:#000; border:none; padding:15px 30px; border-radius:4px; font-family:'Space Mono', monospace; font-size:16px; cursor:pointer; width:300px;">
            ▶ START TRAINING
        </button>
    </div>
    """, unsafe_allow_html=True)
    
    train_button = st.button("Start Training", key="start_training", label_visibility="collapsed")
    
    # Training visualization and results
    if train_button:
        with st.spinner("Preparing training environment..."):
            time.sleep(1)
            
            # Progress bar with TE styling
            progress_placeholder = st.empty()
            terminal_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Current time
            current_time = "2025-07-09 21:15:14"
            
            # Terminal output
            log_lines = [
                f"[{current_time}] Initializing training for {env_name} with {algorithm}",
                f"[{current_time}] Creating environment...",
                f"[{current_time}] Creating agent...",
                f"[{current_time}] Setting up experiment with {training_steps} steps"
            ]
            
            terminal_placeholder.markdown(f"""
            <div class="te-terminal">
            {'<br>'.join(log_lines)}
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate training progress
            total_updates = 20
            for i in range(total_updates + 1):
                progress_pct = i / total_updates * 100
                steps_completed = int(training_steps * (i / total_updates))
                
                # Update progress bar
                progress_html = f"""
                <div style="width: 100%; background-color: #1A1A1A; height: 24px; border-radius: 2px; overflow: hidden; margin: 1em 0; position: relative;">
                    <div style="width: {progress_pct}%; background-color: #FF9900; height: 100%;"></div>
                    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-family: 'Space Mono', monospace; font-size: 0.8em;">
                        {steps_completed} / {training_steps} STEPS ({progress_pct:.1f}%)
                    </div>
                </div>
                """
                progress_placeholder.markdown(progress_html, unsafe_allow_html=True)
                
                time.sleep(0.2)

# VISUALIZATION PAGE
def show_visualization_page():
    st.markdown("<h2>VISUALIZATION</h2>", unsafe_allow_html=True)
    
    # Visualization types
    viz_type = st.selectbox(
        "VISUALIZATION TYPE",
        ["Learning Curves", "Agent Behavior", "Network Architecture", "Value Functions"],
        key="viz_type"
    )
    
    if viz_type == "Learning Curves":
        st.markdown("""
        <div class="te-module" data-title="LEARNING CURVES">
            <p>Visualize and compare learning curves from different experiments.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Select experiments to visualize
        st.markdown("<h3>SELECT EXPERIMENTS</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mock experiments
            experiments = [
                "SAC-Pendulum-v1-2025-07-08",
                "PPO-CartPole-v1-2025-07-08",
                "DQN-LunarLander-v2-2025-07-07",
                "DDPG-BipedalWalker-v3-2025-07-06",
                "SAC-HalfCheetah-v4-2025-07-05"
            ]
            
            selected_exps = st.multiselect(
                "EXPERIMENTS",
                experiments,
                default=[experiments[0], experiments[1]],
                key="selected_exps"
            )
        
        with col2:
            # Metrics to visualize
            metrics = ["Average Reward", "Episode Length", "Value Loss", "Policy Loss", "Entropy"]
            
            selected_metrics = st.multiselect(
                "METRICS",
                metrics,
                default=["Average Reward"],
                key="selected_metrics"
            )
    
    elif viz_type == "Agent Behavior":
        st.write("Agent behavior visualization will go here")
    
    elif viz_type == "Network Architecture":
        st.write("Network architecture visualization will go here")
    
    elif viz_type == "Value Functions":
        st.write("Value function visualization will go here")

# Main app
def main():
    # Get current page from sidebar
    page, use_rllama = sidebar()
    
    # Display content based on selected page
    if page == "ALGORITHMS":
        show_algorithms_page(use_rllama)
        
    elif page == "ENVIRONMENTS":
        show_environments_page()
        
    elif page == "TRAINING LAB":
        show_training_lab_page()
        
    elif page == "VISUALIZATION":
        show_visualization_page()

# Add custom CSS for Teenage Engineering inspired styling
st.markdown("""
<style>
/* General styling */
body {
    background-color: #0D0D0D;
    color: #FFFFFF;
    font-family: 'Inter', sans-serif;
}

/* TE-inspired module containers */
.te-module {
    background-color: #1A1A1A;
    border-radius: 4px;
    padding: 15px;
    margin-bottom: 20px;
    position: relative;
    border: 1px solid #333333;
}

.te-module::before {
    content: attr(data-title);
    position: absolute;
    top: -10px;
    left: 10px;
    background-color: #1A1A1A;
    padding: 0 5px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #FF9900;
}

/* Controls */
.te-control-panel {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
    background-color: #1A1A1A;
    border-radius: 4px;
    padding: 15px;
    border: 1px solid #333333;
}

.te-knob {
    width: 80px;
    height: 80px;
    position: relative;
    margin-bottom: 10px;
}

.te-knob-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #AAAAAA;
    text-align: center;
    margin-top: 5px;
}

.te-toggle {
    display: inline-block;
    width: 60px;
    height: 30px;
    background-color: #333333;
    border-radius: 15px;
    position: relative;
    cursor: pointer;
    transition: background-color 0.3s;
}

.te-toggle.active {
    background-color: #FF9900;
}

.te-toggle::after {
    content: "";
    position: absolute;
    width: 26px;
    height: 26px;
    border-radius: 13px;
    background-color: #FFFFFF;
    top: 2px;
    left: 2px;
    transition: transform 0.3s;
}

.te-toggle.active::after {
    transform: translateX(30px);
}

/* Metrics */
.te-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 20px 0;
}

.te-metric {
    background-color: #1A1A1A;
    border-radius: 4px;
    padding: 15px;
    flex: 1;
    min-width: 120px;
    text-align: center;
    border: 1px solid #333333;
}

.te-metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 24px;
    color: #FF9900;
    margin-bottom: 5px;
}

.te-metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #AAAAAA;
}

/* Terminal-style output */
.te-terminal {
    background-color: #000000;
    border-radius: 4px;
    padding: 15px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #00FF00;
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 20px;
    border: 1px solid #333333;
}

/* Chart container */
.te-chart-container {
    background-color: #1A1A1A;
    border-radius: 4px;
    padding: 15px;
    margin-bottom: 20px;
    border: 1px solid #333333;
}

/* Button styling */
.te-button {
    background-color: #FF9900;
    color: #000000;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.te-button:hover {
    background-color: #FFAA33;
}

/* Tabs */
.te-tabs {
    display: flex;
    gap: 2px;
    margin-bottom: 20px;
}

.te-tab {
    background-color: #1A1A1A;
    color: #AAAAAA;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    padding: 8px 16px;
    cursor: pointer;
    border: 1px solid #333333;
    border-radius: 4px 4px 0 0;
    border-bottom: none;
}

.te-tab.active {
    background-color: #333333;
    color: #FFFFFF;
}

.te-tab-content {
    background-color: #1A1A1A;
    border-radius: 0 4px 4px 4px;
    padding: 15px;
    border: 1px solid #333333;
    margin-bottom: 20px;
}

/* Code editor */
.te-code-editor {
    background-color: #0D0D0D;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    padding: 15px;
    color: #CCCCCC;
    border: 1px solid #333333;
    overflow-x: auto;
}

/* Tags */
.te-tag {
    display: inline-block;
    background-color: #333333;
    color: #FFFFFF;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 4px 8px;
    border-radius: 2px;
    margin-right: 5px;
}

/* Footer */
.te-footer {
    background-color: #1A1A1A;
    border-top: 1px solid #333333;
    padding: 20px;
    text-align: center;
    margin-top: 40px;
    font-size: 12px;
    color: #AAAAAA;
}

/* Override Streamlit's defaults */
.stApp {
    background-color: #0D0D0D;
}

.stSelectbox > div > div, .stSlider > div > div {
    background-color: #1A1A1A;
    color: #FFFFFF;
}

h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF;
    font-family: 'Space Mono', monospace;
}

.stTextInput > div > div > input {
    color: #FFFFFF;
    background-color: #1A1A1A;
}
</style>
""", unsafe_allow_html=True)

# Import necessary fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Add footer
st.markdown('<div class="te-footer">', unsafe_allow_html=True)
st.markdown(f"""
<div style="font-family:'Space Mono', monospace; font-size:1.2em; color:#FF9900; margin-bottom:10px;">RLlama</div>
<div>Reinforcement Learning Laboratory // Teenage Engineering Edition</div>
<div style="margin-top:15px;">
    <a href="https://github.com/yourusername/rllama" style="color:#FF9900; text-decoration:none; margin:0 10px;">GITHUB</a> |
    <a href="https://pypi.org/project/rllama/" style="color:#FF9900; text-decoration:none; margin:0 10px;">PYPI</a> |
    <a href="https://rllama.readthedocs.io/" style="color:#FF9900; text-decoration:none; margin:0 10px;">DOCS</a>
</div>
<div style="margin-top:20px; font-size:0.8em;">SYSTEM TIME: 2025-07-09 21:19:05</div>
<div style="margin-top:5px; font-size:0.8em;">USER: ch33nchan</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
