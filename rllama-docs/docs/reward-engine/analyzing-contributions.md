---
id: analyzing-contributions
title: "Analyzing Component Contributions"
sidebar_label: "Analyzing Contributions"
slug: /reward-engine/analyzing-contributions
---

# Analyzing Component Contributions

One of the key benefits of RLlama's modular approach is the ability to analyze exactly how each component contributes to the total reward. This transparency is invaluable for debugging, understanding agent behavior, and refining reward systems.

## Getting Component Contributions

After computing a reward, you can see how much each component contributed:

```python
# Compute reward
reward = engine.compute(context)

# Get the contributions
contributions = engine.get_last_contributions()
print(f"Total reward: {reward}")
print(f"Component contributions: {contributions}")

# Example output:
# Total reward: 2.35
# Component contributions: {'LengthReward': 0.85, 'DiversityReward': 1.5}
```

This shows that the total reward of 2.35 came from:
- LengthReward: 0.85 (36% of total)
- DiversityReward: 1.5 (64% of total)

## Visualizing Component Contributions

For better understanding, you can visualize these contributions:

```python
import matplotlib.pyplot as plt
import numpy as np

# Get contributions
contributions = engine.get_last_contributions()

# Plot as a bar chart
components = list(contributions.keys())
values = [contributions[comp] for comp in components]

plt.figure(figsize=(10, 6))
bars = plt.bar(components, values)

# Color bars based on positive/negative contribution
for i, v in enumerate(values):
    color = 'green' if v >= 0 else 'red'
    bars[i].set_color(color)

plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.title('Reward Component Contributions')
plt.ylabel('Contribution')
plt.tight_layout()
plt.show()
```

This creates a bar chart showing each component's contribution, with positive contributions in green and negative contributions in red.

## Tracking Contributions Over Time

To understand how component contributions change over time:

```python
import matplotlib.pyplot as plt
import numpy as np

# Lists to store data
rewards = []
component_history = {}

# During training/evaluation
for step in range(100):
    # (Perform your agent's action, get state, etc.)
    
    # Compute reward
    reward = engine.compute(context)
    rewards.append(reward)
    
    # Track contributions
    contributions = engine.get_last_contributions()
    for component, value in contributions.items():
        if component not in component_history:
            component_history[component] = []
        component_history[component].append(value)

# Plot the contribution history
plt.figure(figsize=(12, 6))

# Plot each component's contribution
for component, values in component_history.items():
    plt.plot(values, label=component)

# Plot total reward
plt.plot(rewards, 'k--', linewidth=2, label='Total Reward')

plt.xlabel('Step')
plt.ylabel('Reward/Contribution')
plt.title('Reward Components Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

This creates a line chart showing how each component's contribution evolves over time, along with the total reward.

## Using the RewardVisualizer

RLlama provides a built-in `RewardVisualizer` class for common visualizations:

```python
from rllama.visualization import RewardVisualizer

# Create a visualizer for our engine
visualizer = RewardVisualizer(engine)

# After computing rewards for a sequence
visualizer.add_step_data(reward, engine.get_last_contributions())

# After collecting data for multiple steps
visualizer.plot_reward_history()
visualizer.plot_component_contributions()
visualizer.plot_component_percentages()
```

## Analyzing Relative Importance

To understand the relative importance of each component:

```python
import pandas as pd

# Create lists for analysis
steps = list(range(len(rewards)))
data = {'step': steps, 'total_reward': rewards}

# Add component data
for component, values in component_history.items():
    data[component] = values

# Create DataFrame
df = pd.DataFrame(data)

# Calculate descriptive statistics
stats = df.describe()
print("Component Statistics:")
print(stats)

# Calculate correlations between components and total reward
correlations = df.corr()['total_reward'].drop('total_reward')
print("\nCorrelations with total reward:")
print(correlations)

# Calculate percentage contribution
avg_contributions = {component: np.mean(np.abs(values)) for component, values in component_history.items()}
total_contribution = sum(avg_contributions.values())
percentages = {component: (value / total_contribution) * 100 for component, value in avg_contributions.items()}

print("\nAverage percentage contribution:")
for component, percentage in percentages.items():
    print(f"{component}: {percentage:.1f}%")
```

This analysis provides:
1. Descriptive statistics for each component
2. Correlations between components and total reward
3. Average percentage contribution of each component

## Creating Component Contribution Heatmaps

For more complex scenarios, a heatmap can show interactions between components:

```python
import seaborn as sns

# Calculate correlation matrix
corr_matrix = df.corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Component Correlation Matrix')
plt.tight_layout()
plt.show()
```

This creates a heatmap showing correlations between all components and the total reward, helping identify which components tend to activate together.

## Component Contribution Breakdown by Episodes

For episodic tasks, you can analyze contributions by episode:

```python
# Assuming you have episode data
episode_rewards = []
episode_contributions = []

# For each episode
for episode in range(10):
    rewards = []
    contributions = []
    
    # Reset environment
    state = env.reset()
    done = False
    
    while not done:
        # (Action selection, etc.)
        
        # Compute reward
        reward = engine.compute(context)
        rewards.append(reward)
        contributions.append(engine.get_last_contributions())
        
        # (Environment step, etc.)
    
    # Calculate episode statistics
    episode_total_reward = sum(rewards)
    episode_rewards.append(episode_total_reward)
    
    # Sum contributions across episode
    episode_contrib = {}
    for step_contrib in contributions:
        for component, value in step_contrib.items():
            if component not in episode_contrib:
                episode_contrib[component] = 0
            episode_contrib[component] += value
    
    episode_contributions.append(episode_contrib)

# Plot episode reward breakdown
plt.figure(figsize=(12, 6))
episodes = range(1, len(episode_rewards) + 1)

# Create a stacked bar chart
bottom = np.zeros(len(episodes))
for component in set().union(*episode_contributions):
    values = [contrib.get(component, 0) for contrib in episode_contributions]
    plt.bar(episodes, values, bottom=bottom, label=component)
    bottom += np.array(values)

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Reward Composition by Episode')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

This creates a stacked bar chart showing how different components contribute to the total reward in each episode.

## Advanced Analysis with Interactive Dashboards

For more sophisticated analysis, you can create interactive dashboards:

```python
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd

# Convert data to DataFrame
df_steps = pd.DataFrame(data)

# Create interactive line chart
fig = px.line(df_steps, x='step', y=df_steps.columns[1:],
              title='Reward Component Contributions Over Time')
fig.update_layout(xaxis_title='Step', yaxis_title='Reward/Contribution',
                  legend_title='Component')
fig.show()

# Create interactive bar chart for episode data
episode_df = pd.DataFrame(episode_contributions)
episode_df['episode'] = episodes
episode_df['total'] = episode_rewards

fig = px.bar(episode_df, x='episode', y=episode_df.columns[:-2],
             title='Component Contributions by Episode')
fig.update_layout(xaxis_title='Episode', yaxis_title='Contribution',
                  barmode='group')
fig.show()
```

## Identifying Problematic Components

Analysis can help identify components that might be causing issues:

```python
# Calculate component variability
component_variability = {component: np.std(values) for component, values in component_history.items()}
print("Component variability (standard deviation):")
for component, std in component_variability.items():
    print(f"{component}: {std:.4f}")

# Identify components with extreme values
max_values = {component: np.max(np.abs(values)) for component, values in component_history.items()}
print("\nMaximum absolute values:")
for component, max_val in sorted(max_values.items(), key=lambda x: x[1], reverse=True):
    print(f"{component}: {max_val:.4f}")

# Identify components that frequently output zeros
zero_percentages = {}
for component, values in component_history.items():
    zeros = sum(1 for v in values if abs(v) < 1e-10)
    zero_percentages[component] = (zeros / len(values)) * 100

print("\nPercentage of zero outputs:")
for component, percentage in sorted(zero_percentages.items(), key=lambda x: x[1], reverse=True):
    print(f"{component}: {percentage:.1f}%")
```

This analysis helps identify:
1. Components with high variability
2. Components that produce extreme values
3. Components that frequently output zeros (potentially inactive)

## Exporting Analysis for Reporting

You can export your analysis results for reporting:

```python
# Export contribution data to CSV
pd.DataFrame(data).to_csv('reward_contributions.csv', index=False)

# Export summary statistics
pd.DataFrame(stats).to_csv('reward_statistics.csv')

# Export visualizations
plt.figure(figsize=(12, 6))
for component, values in component_history.items():
    plt.plot(values, label=component)
plt.plot(rewards, 'k--', linewidth=2, label='Total Reward')
plt.xlabel('Step')
plt.ylabel('Reward/Contribution')
plt.title('Reward Components Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reward_analysis.png', dpi=300, bbox_inches='tight')
```

## Summary

Component contribution analysis provides:

1. **Transparency**: Understand exactly what behaviors are being rewarded
2. **Debugging**: Identify components that may be causing unexpected behavior
3. **Optimization**: Focus on the most important components for improvement
4. **Insight**: Gain deeper understanding of your agent's learning process

By leveraging these analysis techniques, you can create more effective, interpretable, and debuggable reward systems for your reinforcement learning agents.
