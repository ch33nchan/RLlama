import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import sys
import os
import logging

# Add RLlama to path - Fix the path issue
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import RLlama components
from rllama.integration.trl_wrapper import TRLRllamaRewardProcessor, TRLRllamaCallback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_dataset(size: int = 50) -> Dataset:
    """Create a more diverse sample dataset"""
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Describe the benefits of renewable energy sources.",
        "How does photosynthesis work in plants?",
        "What are the main causes of climate change?",
        "Explain quantum computing for beginners.",
        "Write a recipe for chocolate chip cookies.",
        "Describe the history of the internet.",
        "What makes a good leader?",
        "Explain the water cycle.",
        "How do vaccines work?",
        "Describe the solar system.",
        "What is artificial intelligence?",
        "Explain how computers process information.",
        "Write about the importance of education.",
        "Describe different types of clouds.",
        "How does the human heart work?",
        "What are the benefits of exercise?",
        "Explain the concept of gravity.",
        "Describe the process of photosynthesis."
    ]
    
    # Repeat and shuffle to get desired size
    import random
    extended_prompts = (prompts * (size // len(prompts) + 1))[:size]
    random.shuffle(extended_prompts)
    
    return Dataset.from_dict({"query": extended_prompts})

def main():
    """Enhanced TRL training example with RLlama"""
    
    print("🦙 Advanced RLlama + TRL Training Example")
    print("=" * 50)
    
    # Configuration
    model_name = "gpt2"
    config_path = os.path.join(os.path.dirname(__file__), "rllama_config_trl.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if config exists, create basic one if not
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, creating basic config...")
        basic_config = """
composer:
  components:
    - type: "CoherenceReward"
      weight: 1.0
    - type: "HelpfulnessReward"
      weight: 0.8
    - type: "DiversityReward"
      weight: 0.6
    - type: "ToxicityReward"
      weight: 1.0

shaper:
  normalization_method: "standard"
"""
        with open(config_path, 'w') as f:
            f.write(basic_config)
        print(f"✅ Created basic config at {config_path}")
    
    print(f"Using device: {device}")
    print(f"RLlama config: {config_path}")
    print(f"Model: {model_name}")
    
    # Initialize RLlama processor
    print("\n1. Initializing RLlama processor...")
    try:
        processor = TRLRllamaRewardProcessor(config_path, device=str(device))
        print("✅ RLlama processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RLlama processor: {e}")
        return
    
    # Initialize callback
    callback = TRLRllamaCallback(processor)
    
    # Load tokenizer and model
    print("\n2. Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with value head for PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        model.to(device)
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create dataset
    print("\n3. Creating training dataset...")
    dataset = create_sample_dataset(30)  # Smaller dataset for testing
    
    def tokenize_function(examples):
        return tokenizer(examples["query"], 
                        truncation=True, 
                        padding=True, 
                        max_length=64,  # Smaller for testing
                        return_tensors="pt")
    
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # PPO Configuration
    print("\n4. Setting up PPO training...")
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1.41e-5,
        batch_size=4,  # Smaller batch size
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=2,  # Fewer epochs for testing
        seed=42,
        steps=5,  # Just 5 steps for testing
    )
    
    # Initialize PPO trainer
    try:
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset
        )
        print("✅ PPO trainer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PPO trainer: {e}")
        return
    
    print(f"Training for {ppo_config.steps} steps with batch size {ppo_config.batch_size}")
    
    # Training loop with enhanced monitoring
    print("\n5. Starting training loop...")
    print("-" * 50)
    
    generation_kwargs = {
        "max_new_tokens": 30,  # Shorter responses for testing
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    try:
        for step, batch in enumerate(ppo_trainer.dataloader):
            if step >= ppo_config.steps:
                break
            
            query_tensors = batch["input_ids"]
            
            # Generate responses
            with torch.no_grad():
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )
            
            # Decode for reward computation
            queries = [tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors]
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
            
            print(f"\nStep {step + 1}/{ppo_config.steps}:")
            print(f"Generated {len(responses)} responses")
            
            # Compute RLlama rewards with batch metadata
            batch_metadata = {
                'step': step + 1,
                'batch_size': len(queries),
                'model_name': model_name,
                'generation_params': generation_kwargs
            }
            
            rewards = processor.compute_rewards(queries, responses, batch_metadata)
            
            # Convert to tensors for PPO
            scores = [torch.tensor(float(r), dtype=torch.float32) for r in rewards]
            
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, scores)
            
            # Enhanced logging
            mean_reward = sum(rewards) / len(rewards)
            mean_score = stats.get('ppo/mean_scores', 0.0)
            kl_div = stats.get('objective/kl', 0.0)
            
            print(f"  Mean RLlama reward: {mean_reward:.4f}")
            print(f"  Mean PPO score: {mean_score:.4f}")
            print(f"  KL divergence: {kl_div:.4f}")
            
            # Show sample interaction
            if step < 2:  # Show first few examples
                print(f"  Sample Query: {queries[0][:80]}...")
                print(f"  Sample Response: {responses[0][:80]}...")
                print(f"  Sample Reward: {rewards[0]:.4f}")
            
            # Callback
            callback.on_step_end(ppo_trainer, step + 1, stats)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Training complete
    print("\n" + "=" * 50)
    print("🎉 Training completed!")
    
    # Final analysis
    print("\n6. Final Analysis:")
    print("-" * 30)
    
    try:
        summary = processor.get_training_summary()
        print(f"Total steps: {summary['total_steps']}")
        print(f"Total samples processed: {summary['total_samples']}")
        print(f"Processing speed: {summary['performance_metrics']['samples_per_second']:.2f} samples/sec")
        
        print(f"\nReward Statistics:")
        raw_stats = summary['overall_statistics']['raw_rewards']
        shaped_stats = summary['overall_statistics']['shaped_rewards']
        print(f"  Raw rewards: {raw_stats['mean']:.4f} ± {raw_stats['std']:.4f}")
        print(f"  Shaped rewards: {shaped_stats['mean']:.4f} ± {shaped_stats['std']:.4f}")
        
        print(f"\nTop Component Contributions:")
        components = summary['component_averages']
        sorted_components = sorted(components.items(), key=lambda x: abs(x[1]), reverse=True)
        for comp_name, avg_contrib in sorted_components[:5]:
            print(f"  {comp_name}: {avg_contrib:.4f}")
        
        # Export results
        print("\n7. Exporting results...")
        processor.export_training_data("advanced_training_analysis.json")
        callback.on_training_end(ppo_trainer)
        
        print("\n✅ All files exported successfully!")
        print("Files created:")
        print("  - advanced_training_analysis.json")
        print("  - final_rllama_analysis.json")
        print("  - trl_training_logs.json")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()