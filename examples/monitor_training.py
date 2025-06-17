import json
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import argparse

def load_training_data(filepath: str) -> Dict[str, Any]:
    """Load training data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filepath} not found. Run training example first.")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filepath}")
        return {}

def create_training_plots(data: Dict[str, Any], output_dir: str = "plots"):
    """Create comprehensive training plots"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if 'detailed_stats' not in data:
        print("No detailed stats found in training data")
        return
    
    stats = data['detailed_stats']
    
    # Extract data for plotting
    steps = [s['step'] for s in stats]
    raw_means = [s['raw_rewards']['mean'] for s in stats]
    shaped_means = [s['shaped_rewards']['mean'] for s in stats]
    
    # Plot 1: Reward progression
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(steps, raw_means, label='Raw Rewards', marker='o')
    plt.plot(steps, shaped_means, label='Shaped Rewards', marker='s')
    plt.xlabel('Training Step')
    plt.ylabel('Mean Reward')
    plt.title('Reward Progression')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Component contributions
    plt.subplot(2, 2, 2)
    if 'component_trends' in data['training_summary']:
        trends = data['training_summary']['component_trends']
        for comp_name, values in list(trends.items())[:5]:  # Top 5 components
            plt.plot(range(len(values)), values, label=comp_name, marker='.')
        plt.xlabel('Training Step')
        plt.ylabel('Component Contribution')
        plt.title('Component Trends')
        plt.legend()
        plt.grid(True)
    
    # Plot 3: Reward distribution
    plt.subplot(2, 2, 3)
    all_raw = [s['raw_rewards']['mean'] for s in stats]
    all_shaped = [s['shaped_rewards']['mean'] for s in stats]
    plt.hist(all_raw, alpha=0.5, label='Raw', bins=20)
    plt.hist(all_shaped, alpha=0.5, label='Shaped', bins=20)
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    
    # Plot 4: Performance metrics
    plt.subplot(2, 2, 4)
    batch_sizes = [s['batch_size'] for s in stats]
    plt.plot(steps, batch_sizes, marker='o')
    plt.xlabel('Training Step')
    plt.ylabel('Batch Size')
    plt.title('Batch Size Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to {output_dir}/training_analysis.png")

def print_summary(data: Dict[str, Any]):
    """Print training summary"""
    if 'training_summary' not in data:
        print("No training summary found")
        return
    
    summary = data['training_summary']
    
    print("🦙 RLlama Training Summary")
    print("=" * 40)
    print(f"Total Steps: {summary.get('total_steps', 0)}")
    print(f"Total Samples: {summary.get('total_samples', 0)}")
    print(f"Processing Speed: {summary.get('performance_metrics', {}).get('samples_per_second', 0):.2f} samples/sec")
    
    print("\nReward Statistics:")
    overall = summary.get('overall_statistics', {})
    raw_stats = overall.get('raw_rewards', {})
    shaped_stats = overall.get('shaped_rewards', {})
    print(f"  Raw: {raw_stats.get('mean', 0):.4f} ± {raw_stats.get('std', 0):.4f}")
    print(f"  Shaped: {shaped_stats.get('mean', 0):.4f} ± {shaped_stats.get('std', 0):.4f}")
    
    print("\nTop Components:")
    components = summary.get('component_averages', {})
    sorted_comps = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)
    for comp_name, avg in sorted_comps[:5]:
        print(f"  {comp_name}: {avg:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Monitor RLlama training progress")
    parser.add_argument("--file", default="advanced_training_analysis.json", 
                       help="Training data file to analyze")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate plots")
    parser.add_argument("--watch", action="store_true", 
                       help="Watch file for changes")
    
    args = parser.parse_args()
    
    if args.watch:
        print(f"Watching {args.file} for changes...")
        last_modified = 0
        
        while True:
            try:
                import os
                current_modified = os.path.getmtime(args.file)
                
                if current_modified > last_modified:
                    print(f"\n[{time.strftime('%H:%M:%S')}] File updated, reloading...")
                    data = load_training_data(args.file)
                    if data:
                        print_summary(data)
                        if args.plot:
                            create_training_plots(data)
                    last_modified = current_modified
                
                time.sleep(2)  # Check every 2 seconds
                
            except KeyboardInterrupt:
                print("\nStopped watching.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
    else:
        # Single analysis
        data = load_training_data(args.file)
        if data:
            print_summary(data)
            if args.plot:
                create_training_plots(data)

if __name__ == "__main__":
    main()