# examples/complete_example.py

import os
import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import RLlama components
from rllama import (
    RewardEngine, 
    BayesianRewardOptimizer, 
    RLlamaAgent,
    RewardTracker,
    launch_dashboard
)
from rllama.rewards.vision_rewards import VisualReasoningReward

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer"""
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def optimize_rewards(config_path, dataset):
    """Optimize reward weights using Bayesian optimization"""
    print("Optimizing reward weights...")
    
    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Define parameter space for optimization
    param_space = {
        "FactualityReward__weight": (0.5, 5.0),
        "CoherenceReward__weight": (0.5, 3.0),
        "RelevanceReward__weight": (0.5, 3.0),
        "HelpfulnessReward__weight": (0.5, 3.0),
        "ToxicityPenalty__weight": (1.0, 5.0),
        "DiversityReward__weight": (0.1, 1.5),
    }
    
    # Define evaluation function
    def evaluate_reward_weights(params):
        # Apply parameters to config
        temp_config = base_config.copy()
        
        for key, value in params.items():
            component, param = key.split("__")
            if component in temp_config["shaping_config"]:
                if isinstance(temp_config["shaping_config"][component], dict):
                    temp_config["shaping_config"][component][param] = value
                else:
                    temp_config["shaping_config"][component] = value
        
        # Create temp config file
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump(temp_config, f)
            temp_path = f.name
        
        # Create engine with this config
        engine = RewardEngine(temp_path)
        
        # Evaluate on sample data
        total_reward = 0.0
        samples = dataset["train"].select(range(min(20, len(dataset["train"]))))
        
        for sample in samples:
            context = {
                "prompt": sample["instruction"],
                "response": sample["output"],
                "factuality_score": np.random.uniform(0.7, 1.0),  # Simulate factuality
                "coherence_score": np.random.uniform(0.6, 1.0),   # Simulate coherence
                "relevance_score": np.random.uniform(0.7, 1.0),   # Simulate relevance
                "query_match": np.random.uniform(0.6, 1.0),       # Simulate query match
                "toxicity_score": np.random.uniform(0.0, 0.2)     # Simulate toxicity
            }
            reward = engine.compute_and_log(context)
            total_reward += reward
        
        # Clean up
        os.unlink(temp_path)
        
        # Return average reward
        return total_reward / len(samples)
    
    # Create optimizer
    optimizer = BayesianRewardOptimizer(
        param_space=param_space,
        eval_function=evaluate_reward_weights,
        direction="maximize",
        n_trials=30
    )
    
    # Run optimization
    results = optimizer.optimize(show_progress_bar=True)
    
    # Generate optimized config
    optimized_config = optimizer.generate_config("optimized_reward_config.yaml")
    
    print(f"Optimization complete. Best reward: {results.best_value}")
    print(f"Best parameters: {results.best_params}")
    print(f"Optimized config saved to: optimized_reward_config.yaml")
    
    return optimized_config, results

def run_training_example(model_name, config_path, dataset, output_dir, num_examples=50):
    """Run a training example using RLlama"""
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    # Create agent
    agent = RLlamaAgent(
        model=model,
        tokenizer=tokenizer,
        device=device,
        reward_config=config_path,
        memory_config={"use_memory": True},
        normalize_rewards=True,
        log_dir=output_dir
    )
    
    # Create reward tracker for visualization
    reward_tracker = RewardTracker(log_dir=os.path.join(output_dir, "reward_logs"))
    
    # Select some examples
    examples = dataset["train"].select(range(min(num_examples, len(dataset["train"]))))
    
    # Track metrics
    total_reward = 0
    metrics = {
        "rewards": [],
        "response_lengths": []
    }
    
    # Process examples
    print(f"Processing {len(examples)} examples...")
    for i, example in enumerate(tqdm(examples)):
        prompt = example["instruction"]
        
        # Generate response
        agent.reset()  # Reset agent state for new example
        response = agent.act(prompt)
        
        # Simulate some metrics that would come from a real environment
        factuality = np.random.uniform(0.7, 1.0)
        coherence = np.random.uniform(0.6, 1.0)
        relevance = np.random.uniform(0.7, 1.0)
        
        # Create context for reward calculation
        context = {
            "prompt": prompt,
            "response": response,
            "factuality_score": factuality,
            "coherence_score": coherence,
            "relevance_score": relevance,
            "query_match": relevance * 0.9,
            "helpfulness_score": coherence * 0.8,
            "toxicity_score": max(0, 1 - coherence - 0.3),
            "response_length": len(response)
        }
        
        # Calculate reward
        reward = agent.compute_reward(context)
        
        # Track reward components
        reward_components = {
            "factuality": factuality * 3.0,
            "coherence": coherence * 1.5,
            "relevance": relevance * 2.0,
            "toxicity": -max(0, 1 - coherence - 0.3) * 2.0
        }
        
        # Log rewards for visualization
        reward_tracker.log_rewards(reward_components, reward)
        
        # Update metrics
        total_reward += reward
        metrics["rewards"].append(reward)
        metrics["response_lengths"].append(len(response))
        
        # End episode in reward tracker
        if (i+1) % 5 == 0:
            reward_tracker.end_episode()
            
        # Print occasional samples
        if i % 10 == 0:
            print(f"\nExample {i}:")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Response: {response[:100]}...")
            print(f"Reward: {reward:.4f}")
    
    # Save reward tracking data
    reward_tracker.save("training_rewards")
    
    # Print summary
    print("\nTraining example complete!")
    print(f"Average reward: {total_reward / len(examples):.4f}")
    print(f"Average response length: {np.mean(metrics['response_lengths']):.1f}")
    
    # Optionally launch dashboard
    launch_dashboard(log_dir=output_dir, port=8501)

def run_visual_reasoning_example():
    """Example demonstrating visual reasoning rewards"""
    try:
        from PIL import Image
        import requests
        from io import BytesIO
        
        # Define example
        image_url = "https://farm4.staticflickr.com/3128/2558131503_d71c6bb620_z.jpg"
        query = "What animal is in this image and what is it doing?"
        response = "The image shows a cat that is sleeping on what appears to be a couch or bed."
        
        # Download image
        response_img = requests.get(image_url)
        img = Image.open(BytesIO(response_img.content))
        
        # Create visual reasoning reward
        visual_reward = VisualReasoningReward()
        
        # Calculate reward
        context = {
            "image": img,
            "query": query,
            "response": response
        }
        
        reward = visual_reward.calculate(context)
        
        print("\nVisual Reasoning Example:")
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Reward: {reward:.4f}")
        
    except ImportError:
        print("Could not run visual reasoning example - required libraries not installed.")

def main():
    parser = argparse.ArgumentParser(description="RLlama Complete Example")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--config", type=str, default="configs/llm_reward_config.yaml", help="Reward config path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Run reward optimization")
    parser.add_argument("--visual", action="store_true", help="Run visual reasoning example")
    parser.add_argument("--examples", type=int, default=50, help="Number of examples to process")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using a small dummy dataset instead.")
        dataset = {
            "train": [
                {"instruction": "Explain quantum computing.", "output": "Quantum computing uses quantum bits which can be in multiple states simultaneously."},
                {"instruction": "Write a short poem about mountains.", "output": "Peaks reaching high, touching clouds with grace, mountains stand strong."},
            ] * 25
        }
    
    # Run optimization if requested
    if args.optimize:
        config_path, results = optimize_rewards(args.config, dataset)
    else:
        config_path = args.config
    
    # Run training example
    run_training_example(
        model_name=args.model,
        config_path=config_path,
        dataset=dataset,
        output_dir=args.output_dir,
        num_examples=args.examples
    )
    
    # Run visual example if requested
    if args.visual:
        run_visual_reasoning_example()

if __name__ == "__main__":
    main()