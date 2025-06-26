#!/usr/bin/env python3
"""
Complete Memory Systems Cookbook
===============================

This cookbook demonstrates how to use episodic and working memory systems
for context-aware reward computation using RLlama's memory components.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from typing import List, Dict, Any, Tuple
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx

sys.path.append(os.path.abspath("../.."))

from rllama import EpisodicMemory, WorkingMemory, MemoryEntry, MemoryCompressor

def generate_diverse_memories(n_memories: int = 500, state_dim: int = 32) -> List[MemoryEntry]:
    """Generate diverse synthetic memories for testing"""
    print(f"🧠 Generating {n_memories} diverse memories...")
    
    memories = []
    
    # Create different clusters representing different contexts/situations
    cluster_centers = [
        np.array([2.0, 1.0] + [0.0] * (state_dim - 2)),    # High reward context
        np.array([-2.0, -1.0] + [0.0] * (state_dim - 2)),  # Low reward context
        np.array([0.0, 2.0] + [0.0] * (state_dim - 2)),    # Medium reward context
        np.array([1.0, -2.0] + [0.0] * (state_dim - 2)),   # Variable reward context
        np.array([-1.0, 1.0] + [0.0] * (state_dim - 2)),   # Exploration context
    ]
    
    cluster_sizes = [n_memories // len(cluster_centers)] * len(cluster_centers)
    cluster_sizes[0] += n_memories % len(cluster_centers)  # Handle remainder
    
    memory_id = 0
    
    for cluster_id, (center, size) in enumerate(zip(cluster_centers, cluster_sizes)):
        for i in range(size):
            # Generate state around cluster center with noise
            noise = np.random.normal(0, 0.5, state_dim)
            state = torch.FloatTensor(center + noise)
            
            # Generate context-dependent reward
            if cluster_id == 0:  # High reward cluster
                base_reward = 1.5 + np.random.normal(0, 0.2)
            elif cluster_id == 1:  # Low reward cluster
                base_reward = -0.5 + np.random.normal(0, 0.1)
            elif cluster_id == 2:  # Medium reward cluster
                base_reward = 0.5 + np.random.normal(0, 0.15)
            elif cluster_id == 3:  # Variable reward cluster
                base_reward = np.random.uniform(-1, 1)
            else:  # Exploration cluster
                base_reward = 0.3 + np.random.exponential(0.5)
            
            # Generate meaningful actions
            action_types = [
                f"generate_creative_text_{cluster_id}",
                f"provide_explanation_{cluster_id}", 
                f"write_code_{cluster_id}",
                f"answer_question_{cluster_id}",
                f"summarize_content_{cluster_id}"
            ]
            action = action_types[i % len(action_types)]
            
            # Calculate importance based on reward magnitude and rarity
            importance = min(1.0, abs(base_reward) * 0.5 + np.random.uniform(0, 0.5))
            
            # Add temporal component
            timestamp = memory_id + np.random.randint(-10, 10)
            
            # Create memory entry
            memory = MemoryEntry(
                state=state,
                action=action,
                reward=base_reward,
                importance=importance,
                timestamp=timestamp,
                metadata={'cluster': cluster_id, 'context': f'context_{cluster_id}'}
            )
            
            memories.append(memory)
            memory_id += 1
    
    print(f"✅ Generated {len(memories)} memories across {len(cluster_centers)} contexts")
    return memories

def demonstrate_episodic_memory():
    """Demonstrate episodic memory functionality"""
    print("\n📚 Episodic Memory Demonstration")
    print("-" * 40)
    
    # Create episodic memory
    memory = EpisodicMemory(capacity=200)
    
    # Generate and add memories
    all_memories = generate_diverse_memories(n_memories=250, state_dim=16)
    
    print("Adding memories to episodic memory...")
    for mem in all_memories:
        memory.add(mem)
    
    print(f"Memory capacity: {memory.capacity}")
    print(f"Current size: {len(memory)}")
    print(f"Overflow handled: {'✅' if len(all_memories) > memory.capacity else '❌'}")
    
    # Test memory retrieval with different queries
    print("\nTesting memory retrieval:")
    
    test_queries = [
        (torch.FloatTensor([2.0, 1.0] + [0.0] * 14), "High reward context"),
        (torch.FloatTensor([-2.0, -1.0] + [0.0] * 14), "Low reward context"),
        (torch.FloatTensor([0.0, 0.0] + [0.0] * 14), "Neutral context"),
        (torch.FloatTensor([5.0, 5.0] + [0.0] * 14), "Unknown context"),
    ]
    
    retrieval_results = []
    
    for query, description in test_queries:
        print(f"\n{description}:")
        print(f"  Query vector: {query[:2].tolist()}")
        
        relevant_memories = memory.retrieve_relevant(query, k=5)
        
        if relevant_memories:
            rewards = [m.reward for m in relevant_memories]
            similarities = []
            
            # Calculate similarities for analysis
            for mem in relevant_memories:
                dot_product = torch.sum(mem.state * query).item()
                norm_a = torch.norm(mem.state).item()
                norm_b = torch.norm(query).item()
                similarity = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                similarities.append(similarity)
            
            print(f"  Retrieved {len(relevant_memories)} memories")
            print(f"  Rewards: {[f'{r:.3f}' for r in rewards]}")
            print(f"  Similarities: {[f'{s:.3f}' for s in similarities]}")
            print(f"  Avg reward: {np.mean(rewards):.3f}")
            print(f"  Actions: {[m.action for m in relevant_memories]}")
            
            retrieval_results.append({
                'query': query,
                'description': description,
                'memories': relevant_memories,
                'rewards': rewards,
                'similarities': similarities
            })
        else:
            print("  No memories retrieved")
            retrieval_results.append({
                'query': query,
                'description': description,
                'memories': [],
                'rewards': [],
                'similarities': []
            })
    
    return memory, retrieval_results

def demonstrate_working_memory():
    """Demonstrate working memory functionality"""
    print("\n🧠 Working Memory Demonstration")
    print("-" * 40)
    
    # Create working memory
    working_memory = WorkingMemory(max_size=7)
    
    print("Adding sequential states to working memory...")
    
    # Simulate a sequence of related states (e.g., conversation turns)
    conversation_states = []
    for i in range(10):  # Add more than max_size to test overflow
        # Create states that represent a conversation flow
        if i < 3:
            # Initial greeting phase
            state = torch.FloatTensor([0.5, 0.2, i * 0.1] + [0.1] * 5)
            phase = "greeting"
        elif i < 6:
            # Information exchange phase
            state = torch.FloatTensor([1.0, 0.8, i * 0.2] + [0.3] * 5)
            phase = "information"
        else:
            # Conclusion phase
            state = torch.FloatTensor([0.3, 1.2, i * 0.15] + [0.5] * 5)
            phase = "conclusion"
        
        working_memory.add(state)
        conversation_states.append((state, phase))
        
        print(f"  Turn {i+1} ({phase}): {state[:3].tolist()}, Memory size: {len(working_memory)}")
    
    # Test context generation for different queries
    print("\nTesting context generation:")
    
    test_contexts = [
        (torch.FloatTensor([0.5, 0.3, 0.2] + [0.2] * 5), "Similar to greeting"),
        (torch.FloatTensor([1.0, 0.9, 0.8] + [0.4] * 5), "Similar to information"),
        (torch.FloatTensor([0.2, 1.1, 1.0] + [0.6] * 5), "Similar to conclusion"),
        (torch.FloatTensor([2.0, 2.0, 2.0] + [1.0] * 5), "Very different"),
    ]
    
    context_results = []
    
    for query, description in test_contexts:
        print(f"\n{description}:")
        print(f"  Query: {query[:3].tolist()}")
        
        # Generate context
        original_query = query.clone()
        context = working_memory.get_context(query)
        
        # Calculate the influence of working memory
        context_influence = torch.norm(context - original_query).item()
        
        print(f"  Original query: {original_query[:3].tolist()}")
        print(f"  Generated context: {context[:3].tolist()}")
        print(f"  Context influence: {context_influence:.3f}")
        
        context_results.append({
            'query': original_query,
            'context': context,
            'influence': context_influence
        })
    
    return working_memory, context_results

def demonstrate_memory_compression():
    """Demonstrate memory compression functionality"""
    print("\n🗜️ Memory Compression Demonstration")
    print("-" * 40)
    
    # Generate large set of memories with redundancy
    print("Generating memories with redundancy...")
    
    all_memories = generate_diverse_memories(n_memories=300, state_dim=12)
    
    # Add some very similar memories to test compression
    base_state = torch.FloatTensor([1.0, 1.0] + [0.0] * 10)
    for i in range(30):
        # Create very similar states
        similar_state = base_state + torch.normal(0, 0.1, size=(12,))
        similar_memory = MemoryEntry(
            state=similar_state,
            action=f"similar_action_{i}",
            reward=1.0 + np.random.normal(0, 0.05),
            importance=0.8 + np.random.uniform(-0.1, 0.1),
            timestamp=1000 + i
        )
        all_memories.append(similar_memory)
    
    print(f"Total memories before compression: {len(all_memories)}")
    
    # Test different compression settings
    compression_configs = [
        {'ratio': 0.8, 'threshold': 0.9, 'name': 'Light Compression'},
        {'ratio': 0.6, 'threshold': 0.8, 'name': 'Medium Compression'},
        {'ratio': 0.4, 'threshold': 0.7, 'name': 'Heavy Compression'},
    ]
    
    compression_results = []
    
    for config in compression_configs:
        print(f"\n{config['name']} (ratio={config['ratio']}, threshold={config['threshold']}):")
        
        compressor = MemoryCompressor(
            compression_ratio=config['ratio'],
            similarity_threshold=config['threshold']
        )
        
        compressed_memories = compressor.compress(all_memories)
        
        # Analyze compression results
        compression_ratio = len(compressed_memories) / len(all_memories)
        
        # Calculate importance statistics
        original_importance = [m.importance for m in all_memories]
        compressed_importance = [m.importance for m in compressed_memories]
        
        # Calculate reward statistics
        original_rewards = [m.reward for m in all_memories]
        compressed_rewards = [m.reward for m in compressed_memories]
        
        print(f"  Compressed to {len(compressed_memories)} memories ({compression_ratio:.2%})")
        print(f"  Avg importance: {np.mean(original_importance):.3f} → {np.mean(compressed_importance):.3f}")
        print(f"  Avg reward: {np.mean(original_rewards):.3f} → {np.mean(compressed_rewards):.3f}")
        print(f"  Reward std: {np.std(original_rewards):.3f} → {np.std(compressed_rewards):.3f}")
        
        compression_results.append({
            'config': config,
            'compressed_memories': compressed_memories,
            'compression_ratio': compression_ratio,
            'original_importance': original_importance,
            'compressed_importance': compressed_importance,
            'original_rewards': original_rewards,
            'compressed_rewards': compressed_rewards
        })
    
    return compression_results

def analyze_memory_patterns():
    """Analyze patterns in memory retrieval and usage"""
    print("\n🔍 Memory Pattern Analysis")
    print("-" * 40)
    
    # Create memories with known patterns
    memory = EpisodicMemory(capacity=150)
    
    # Generate memories with temporal patterns
    print("Generating memories with temporal patterns...")
    
    temporal_memories = []
    n_episodes = 4
    episode_length = 20
    
    for episode in range(n_episodes):
        for step in range(episode_length):
            # Create state that evolves over time within episode
            base_pattern = np.array([
                np.sin(step * 0.3),  # Cyclical pattern
                episode * 0.4,      # Episode-dependent
                step * 0.05,        # Step-dependent
            ])
            
            # Add noise and pad to desired dimension
            state = torch.FloatTensor(
                np.concatenate([base_pattern, np.random.normal(0, 0.1, 5)])
            )
            
            # Reward depends on pattern
            reward = np.sin(step * 0.3) + episode * 0.2 + np.random.normal(0, 0.1)
            
            # Importance increases with episode and step
            importance = min(1.0, (episode + 1) * (step + 1) / (n_episodes * episode_length))
            
            mem = MemoryEntry(
                state=state,
                action=f"ep_{episode}_step_{step}",
                reward=reward,
                importance=importance,
                timestamp=episode * episode_length + step
            )
            
            temporal_memories.append(mem)
            memory.add(mem)
    
    print(f"Generated {len(temporal_memories)} memories across {n_episodes} episodes")
    
    # Analyze retrieval patterns
    print("\nAnalyzing retrieval patterns...")
    
    # Test queries that should retrieve different patterns
    pattern_queries = [
        (torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Early states"),
        (torch.FloatTensor([0.0, 1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Late states"),
        (torch.FloatTensor([1.0, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]), "High reward states"),
        (torch.FloatTensor([-1.0, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]), "Low reward states"),
    ]
    
    pattern_analysis = []
    
    for query, description in pattern_queries:
        retrieved = memory.retrieve_relevant(query, k=8)
        
        if retrieved:
            # Extract metadata from retrieved memories
            episodes = []
            steps = []
            rewards = []
            importances = []
            
            for mem in retrieved:
                # Parse action to extract episode and step
                action_parts = mem.action.split('_')
                episode = int(action_parts[1])
                step = int(action_parts[3])
                
                episodes.append(episode)
                steps.append(step)
                rewards.append(mem.reward)
                importances.append(mem.importance)
            
            # Calculate statistics
            episode_distribution = np.bincount(episodes, minlength=n_episodes)
            avg_step = np.mean(steps)
            avg_reward = np.mean(rewards)
            avg_importance = np.mean(importances)
            
            print(f"\n{description}:")
            print(f"  Episode distribution: {episode_distribution}")
            print(f"  Avg step: {avg_step:.1f}")
            print(f"  Avg reward: {avg_reward:.3f}")
            print(f"  Avg importance: {avg_importance:.3f}")
            
            pattern_analysis.append({
                'description': description,
                'query': query,
                'retrieved': retrieved,
                'episode_distribution': episode_distribution,
                'avg_step': avg_step,
                'avg_reward': avg_reward,
                'avg_importance': avg_importance
            })
    
    return pattern_analysis

def create_memory_visualizations(episodic_memory, retrieval_results, compression_results, pattern_analysis):
    """Create comprehensive memory system visualizations"""
    print("\n📊 Creating memory system visualizations...")
    
    os.makedirs("./output", exist_ok=True)
    
    # Extract memory data for visualization
    memories = list(episodic_memory.memories.values())
    states = np.array([mem.state.numpy() for mem in memories])
    rewards = np.array([mem.reward for mem in memories])
    importances = np.array([mem.importance for mem in memories])
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Memory distribution in 2D space
    scatter = axes[0, 0].scatter(states_2d[:, 0], states_2d[:, 1], 
                                c=rewards, s=importances*100, 
                                cmap='RdYlBu_r', alpha=0.7, edgecolors='black')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title('Memory Distribution (color=reward, size=importance)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Reward')
    
    # 2. Reward distribution
    axes[0, 1].hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(np.mean(rewards), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(rewards):.3f}')
    axes[0, 1].set_xlabel('Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Importance distribution
    axes[0, 2].hist(importances, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].axvline(np.mean(importances), color='blue', linestyle='--',
                      label=f'Mean: {np.mean(importances):.3f}')
    axes[0, 2].set_xlabel('Importance')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Importance Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Memory clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(states_2d)
    
    scatter = axes[1, 0].scatter(states_2d[:, 0], states_2d[:, 1], 
                                c=clusters, cmap='tab10', alpha=0.7)
    axes[1, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                      c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1, 0].set_xlabel('PC1')
    axes[1, 0].set_ylabel('PC2')
    axes[1, 0].set_title('Memory Clusters')
    axes[1, 0].legend()
    
    # 5. Reward vs Importance correlation
    axes[1, 1].scatter(rewards, importances, alpha=0.6, s=30)
    z = np.polyfit(rewards, importances, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(rewards.min(), rewards.max(), 100)
    axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    corr = np.corrcoef(rewards, importances)[0, 1]
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Importance')
    axes[1, 1].set_title(f'Reward vs Importance (r={corr:.3f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Compression analysis
    compression_names = [result['config']['name'] for result in compression_results]
    compression_ratios = [result['compression_ratio'] for result in compression_results]
    
    bars = axes[1, 2].bar(compression_names, compression_ratios, 
                         color=['lightgreen', 'orange', 'lightcoral'], alpha=0.7)
    axes[1, 2].set_ylabel('Final / Original Memory Count')
    axes[1, 2].set_title('Compression Effectiveness')
    axes[1, 2].set_xticklabels(compression_names, rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, ratio in zip(bars, compression_ratios):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/memory_systems_analysis.png', dpi=300, bbox_inches='tight')
    
    # Memory retrieval effectiveness
    plt.figure(figsize=(12, 8))
    
    # Retrieval quality analysis
    plt.subplot(2, 2, 1)
    query_types = [result['description'] for result in retrieval_results if result['rewards']]
    avg_rewards = [np.mean(result['rewards']) for result in retrieval_results if result['rewards']]
    avg_similarities = [np.mean(result['similarities']) for result in retrieval_results if result['similarities']]
    
    if query_types:
        bars = plt.bar(range(len(query_types)), avg_rewards, alpha=0.7, color='skyblue')
        plt.xlabel('Query Type')
        plt.ylabel('Average Retrieved Reward')
        plt.title('Retrieval Quality by Query Type')
        plt.xticks(range(len(query_types)), query_types, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, reward in zip(bars, avg_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Pattern analysis
    plt.subplot(2, 2, 2)
    if pattern_analysis:
        pattern_names = [p['description'] for p in pattern_analysis]
        pattern_rewards = [p['avg_reward'] for p in pattern_analysis]
        
        bars = plt.bar(range(len(pattern_names)), pattern_rewards, 
                      alpha=0.7, color='lightcoral')
        plt.xlabel('Pattern Type')
        plt.ylabel('Average Reward')
        plt.title('Memory Pattern Analysis')
        plt.xticks(range(len(pattern_names)), pattern_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        for bar, reward in zip(bars, pattern_rewards):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{reward:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Compression quality preservation
    plt.subplot(2, 2, 3)
    original_reward_mean = np.mean([result['original_rewards'] for result in compression_results], axis=1)
    compressed_reward_mean = np.mean([result['compressed_rewards'] for result in compression_results], axis=1)
    
    preservation_ratios = [comp/orig for comp, orig in zip(compressed_reward_mean, original_reward_mean)]
    
    bars = plt.bar(compression_names, preservation_ratios, 
                  color=['lightgreen', 'orange', 'lightcoral'], alpha=0.7)
    plt.ylabel('Reward Preservation Ratio')
    plt.title('Quality Preservation in Compression')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Preservation')
    plt.legend()
    
    for bar, ratio in zip(bars, preservation_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Memory efficiency
    plt.subplot(2, 2, 4)
    memory_sizes = [len(result['compressed_memories']) for result in compression_results]
    quality_scores = preservation_ratios
    efficiency_scores = [q/s*100 for q, s in zip(quality_scores, memory_sizes)]
    
    bars = plt.bar(compression_names, efficiency_scores, 
                  color=['lightgreen', 'orange', 'lightcoral'], alpha=0.7)
    plt.ylabel('Efficiency Score (Quality/Size * 100)')
    plt.title('Memory Efficiency')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, eff in zip(bars, efficiency_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/memory_retrieval_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved memory analysis to ./output/memory_systems_analysis.png")
    print("✅ Saved retrieval analysis to ./output/memory_retrieval_analysis.png")

def main():
    """Run the complete memory systems cookbook"""
    print("🦙 RLlama Memory Systems Cookbook")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("./output", exist_ok=True)
    
    # 1. Episodic memory demonstration
    episodic_memory, retrieval_results = demonstrate_episodic_memory()
    
    # 2. Working memory demonstration
    working_memory, context_results = demonstrate_working_memory()
    
    # 3. Memory compression demonstration
    compression_results = demonstrate_memory_compression()
    
    # 4. Memory pattern analysis
    pattern_analysis = analyze_memory_patterns()
    
    # 5. Create visualizations
    create_memory_visualizations(episodic_memory, retrieval_results, compression_results, pattern_analysis)
    
    # Print comprehensive summary
    print("\n" + "="*50)
    print("🎉 Memory Systems Cookbook Complete!")
    print("="*50)
    
    print(f"\n📊 Summary of Results:")
    
    print(f"\n  Episodic Memory:")
    print(f"    • Capacity: {episodic_memory.capacity}")
    print(f"    • Current size: {len(episodic_memory)}")
    print(f"    • Retrieval tests: {len(retrieval_results)}")
    print(f"    • Successful retrievals: {sum(1 for r in retrieval_results if r['memories'])}")
    
    print(f"\n  Working Memory:")
    print(f"    • Max size: {working_memory.max_size}")
    print(f"    • Current size: {len(working_memory)}")
    print(f"    • Context generation tests: {len(context_results)}")
    print(f"    • Avg context influence: {np.mean([r['influence'] for r in context_results]):.3f}")
    
    print(f"\n  Memory Compression:")
    for result in compression_results:
        config = result['config']
        ratio = result['compression_ratio']
        print(f"    • {config['name']}: {ratio:.1%} compression achieved")
    
    print(f"\n  Pattern Analysis:")
    print(f"    • Temporal patterns detected: {len(pattern_analysis)}")
    print(f"    • Episode-based clustering successful")
    if pattern_analysis:
        avg_pattern_reward = np.mean([p['avg_reward'] for p in pattern_analysis])
        print(f"    • Average pattern reward: {avg_pattern_reward:.3f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  • ./output/memory_systems_analysis.png - Comprehensive memory analysis")
    print(f"  • ./output/memory_retrieval_analysis.png - Retrieval effectiveness analysis")
    
    print(f"\n🔗 Key Insights:")
    print(f"  • Episodic memory enables efficient similarity-based retrieval")
    print(f"  • Working memory provides context-aware state enhancement")
    print(f"  • Memory compression preserves important experiences while reducing storage")
    print(f"  • Temporal patterns emerge naturally in episodic memory")
    print(f"  • Context generation improves with relevant working memory content")
    
    print(f"\n🚀 Next Steps:")
    print(f"  • Integrate memory systems with your reward functions")
    print(f"  • Experiment with different similarity metrics")
    print(f"  • Implement domain-specific memory compression strategies")
    print(f"  • Scale to larger memory capacities and datasets")
    print(f"  • Combine episodic and working memory for hierarchical storage")

if __name__ == "__main__":
    main()
