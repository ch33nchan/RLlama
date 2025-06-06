"""
Complete TRL + RLlama Integration Example
Demonstrates end-to-end reward engineering with PPO training
"""

import os
import sys
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
import logging
import json

# Ensure RLlama is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rllama.integration.trl_wrapper import TRLRllamaRewardProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = "gpt2"  # Can be changed to any compatible model
DATASET_SIZE = 100   # Number of training samples
NUM_TRAIN_STEPS = 10  # Number of PPO training steps
BATCH_SIZE = 4       # Training batch size
MINI_BATCH_SIZE = 2  # PPO mini batch size
PPO_EPOCHS = 4       # PPO epochs per step
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1.41e-5
MAX_LENGTH = 128     # Maximum response length
RLLAMA_CONFIG_PATH = "examples/rllama_config_trl.yaml"

def create_sample_dataset(size: int = 100) -> Dataset:
    """Create a sample dataset for training"""
    queries = [
        "Describe a beautiful sunset.",
        "Explain the concept of gravity.",
        "Write a short story about friendship.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "Describe your favorite season.",
        "Explain the importance of education.",
        "Write about a memorable childhood experience.",
        "What makes a good leader?",
        "Describe the process of making coffee.",
    ] * (size // 10 + 1)
    
    # Truncate to exact size
    queries = queries[:size]
    
    return Dataset.from_dict({"query": queries})

def main():
    """Main training function"""
    logger.info("Starting TRL + RLlama training example")
    
    # --- 1. Initialize Dataset ---
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset(DATASET_SIZE)
    logger.info(f"Created dataset with {len(dataset)} samples")
    
    # --- 2. Initialize Model, Tokenizer ---
    logger.info(f"Loading model and tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token for GPT-2
    
    # PPOConfig sets up the PPO training parameters
    ppo_config = PPOConfig(
        model_name=MODEL_NAME,
        learning_rate=LEARNING_RATE,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        log_with=None,  # "wandb" or None
        tracker_project_name="trl_rllama_example",
        optimize_cuda_cache=True,
    )
    
    # Load model for PPO training
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        peft_config=None,  # Optional: PEFT config for LoRA, etc.
    )
    
    # Ensure model and tokenizer use the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    
    # --- 3. Initialize RLlama Reward Processor ---
    try:
        rllama_processor = TRLRllamaRewardProcessor(
            rllama_config_path=RLLAMA_CONFIG_PATH,
            auto_register_llm_components=True
        )
        logger.info(f"RLlama processor initialized with config: {RLLAMA_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error initializing RLlama processor: {e}")
        logger.error("Please ensure that the config file exists and is properly formatted.")
        sys.exit(1)
    
    # --- 4. Initialize PPOTrainer ---
    try:
        ppo_trainer = PPOTrainer(
            model=model,
            config=ppo_config,
            dataset=dataset,
            tokenizer=tokenizer,
        )
        logger.info("PPO trainer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing PPO trainer: {e}")
        sys.exit(1)
    
    # --- 5. Generation Configuration ---
    generation_kwargs = {
        "max_new_tokens": MAX_LENGTH,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # --- 6. Training Loop ---
    logger.info(f"Starting training for {NUM_TRAIN_STEPS} steps...")
    
    # Storage for analysis
    training_stats = []
    
    for step in tqdm(range(NUM_TRAIN_STEPS), desc="PPO Training Steps"):
        
        # --- Get batch of queries ---
        batch_start = (step * BATCH_SIZE) % len(dataset)
        batch_end = min(batch_start + BATCH_SIZE, len(dataset))
        
        if batch_end <= batch_start:
            batch_start = 0
            batch_end = BATCH_SIZE
        
        query_texts = dataset[batch_start:batch_end]["query"]
        
        # Tokenize queries
        query_tensors = []
        for query in query_texts:
            query_tensor = tokenizer.encode(query, return_tensors="pt").squeeze(0)
            query_tensors.append(query_tensor.to(device))
        
        # --- Generate responses ---
        try:
            response_tensors = ppo_trainer.generate(
                query_tensors, 
                return_prompt=False,
                **generation_kwargs
            )
            
            # Decode responses to text
            response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
            
            logger.info(f"Step {step+1}: Generated {len(response_texts)} responses")
            
        except Exception as e:
            logger.error(f"Error generating responses at step {step}: {e}")
            continue
        
        # --- Compute rewards using RLlama ---
        try:
            rewards = rllama_processor.compute_rewards(
                prompts_text=query_texts,
                responses_text=response_texts,
                model_specific_infos=None
            )
            
            # Convert to tensors for TRL (scalar tensors, not shape [1])
            scores = [torch.tensor(float(r), dtype=torch.float32, device=device) for r in rewards]
            
            logger.info(f"Step {step+1}: Computed rewards - Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
            
        except Exception as e:
            logger.error(f"Error computing RLlama rewards at step {step}: {e}")
            rewards = [0.0 for _ in query_texts]
            scores = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in rewards]
        
        # --- Perform PPO step ---
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, scores)
            
            # Log step information
            step_info = {
                'step': step + 1,
                'mean_ppo_reward': stats.get('ppo/mean_scores', 0.0),
                'objective_kl': stats.get('objective/kl', 0.0),
                'mean_rllama_reward': np.mean(rewards),
                'std_rllama_reward': np.std(rewards),
                'example_query': query_texts[0][:50] + "..." if query_texts else "",
                'example_response': response_texts[0][:50] + "..." if response_texts else "",
                'example_reward': rewards[0] if rewards else 0.0
            }
            
            training_stats.append(step_info)
            
            if step % 1 == 0:  # Log every step for this example
                logger.info(f"\n--- Step {step+1}/{NUM_TRAIN_STEPS} ---")
                logger.info(f"  Mean PPO reward: {step_info['mean_ppo_reward']:.4f}")
                logger.info(f"  Objective/kl: {step_info['objective_kl']:.4f}")
                logger.info(f"  Mean RLlama reward: {step_info['mean_rllama_reward']:.4f}")
                logger.info(f"  Example Query: {step_info['example_query']}")
                logger.info(f"  Example Response: {step_info['example_response']}")
                logger.info(f"  Example Reward: {step_info['example_reward']:.3f}")
                
                # Show component breakdown for first sample
                if step == 0:
                    last_batch_info = rllama_processor.get_last_batch_detailed_infos()
                    if last_batch_info and 'components' in last_batch_info:
                        logger.info("  Component breakdown:")
                        for comp_name, comp_scores in last_batch_info['components'].items():
                            if comp_scores:
                                logger.info(f"    {comp_name}: {comp_scores[0]:.4f}")
            
        except Exception as e:
            logger.error(f"Error during PPO step {step}: {e}")
            continue
        
        # Reset RLlama components periodically if needed
        if (step + 1) % 50 == 0:
            logger.info("Resetting RLlama components")
            rllama_processor.reset_components()
    
    logger.info("\nTraining finished.")
    
    # --- Final Analysis and Visualization ---
    logger.info("\n=== RLlama Analysis ===")
    
    try:
        analysis = rllama_processor.get_component_analysis()
        logger.info(f"Total training steps: {analysis.get('total_steps', 0)}")
        
        logger.info("\nComponent average contributions:")
        avg_rewards = analysis.get('avg_rewards', {})
        for comp_name, avg_reward in avg_rewards.items():
            logger.info(f"  {comp_name}: {avg_reward:.4f}")
        
        # Show overall statistics
        overall_stats = analysis.get('overall_statistics', {})
        if overall_stats:
            raw_stats = overall_stats.get('raw_rewards', {})
            norm_stats = overall_stats.get('normalized_rewards', {})
            
            logger.info(f"\nOverall reward statistics:")
            logger.info(f"  Raw rewards - Mean: {raw_stats.get('mean', 0):.4f}, Std: {raw_stats.get('std', 0):.4f}")
            logger.info(f"  Normalized rewards - Mean: {norm_stats.get('mean', 0):.4f}, Std: {norm_stats.get('std', 0):.4f}")
        
        # Get detailed info from last batch
        last_batch = rllama_processor.get_last_batch_detailed_infos()
        if last_batch:
            logger.info(f"\nLast batch (Step {last_batch.get('step', 'N/A')}):")
            raw_rewards = last_batch.get('raw_rewards', [])
            norm_rewards = last_batch.get('normalized_rewards', [])
            if raw_rewards and norm_rewards:
                logger.info(f"  Raw rewards: {[f'{r:.3f}' for r in raw_rewards[:3]]}...")
                logger.info(f"  Normalized rewards: {[f'{r:.3f}' for r in norm_rewards[:3]]}...")
        
        # Save analysis to file
        analysis_file = "rllama_training_analysis.json"
        rllama_processor.save_analysis(analysis_file)
        logger.info(f"Detailed analysis saved to {analysis_file}")
        
        # Save training statistics
        with open("training_stats.json", "w") as f:
            json.dump(training_stats, f, indent=2)
        logger.info("Training statistics saved to training_stats.json")
        
    except Exception as e:
        logger.error(f"Error in final analysis: {e}")
    
    # --- Optional: Save the model ---
    save_model = True  # Set to True to save the trained model
    if save_model:
        output_dir = "./trl_rllama_finetuned_model"
        try:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main()

