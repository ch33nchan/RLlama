# /Users/cheencheen/Desktop/git/rl/RLlama/examples/trl_rllama_example.py

import os
import sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, pipeline # Removed AutoModelForCausalLMWithValueHead from here
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead # Added AutoModelForCausalLMWithValueHead here
from tqdm import tqdm

# Ensure RLlama is in the Python path
# Assuming 'examples' is a subdirectory of the project root where 'rllama' package resides
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rllama.integration.trl_wrapper import TRLRllamaRewardProcessor
# We need to ensure the LLM components used in the YAML are defined and registered.
# For this example, let's assume they are in rllama.rewards.llm_components
# and that register_llm_components_if_not_present in TRLRllamaRewardProcessor handles them.
# If not, we would need to import and register them manually here or ensure TRLRllamaRewardProcessor does.
# from rllama.rewards.llm_components import CoherenceReward, ConcisionReward, DiversityReward, FactualityReward
# from rllama.utils.config_loader import register_component
# register_component("CoherenceReward", CoherenceReward)
# register_component("ConcisionReward", ConcisionReward)
# register_component("DiversityReward", DiversityReward)
# register_component("FactualityReward", FactualityReward) # May be complex to implement fully

# --- Configuration ---
MODEL_NAME = "lvwerra/gpt2-imdb" # A model fine-tuned for sentiment, good for a simple task
# MODEL_NAME = "gpt2" # Or a base GPT-2 model
RLLAMA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "rllama_config_trl.yaml")
LEARNING_RATE = 1.41e-5
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 4 # Keep small for example
BATCH_SIZE = 16     # Number of prompts to process at once
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GEN_LENGTH = 50 # Max length of generated response
NUM_TRAIN_STEPS = 10 # Number of PPO steps for the example

# --- 1. Define a simple dataset ---
# Let's create a simple instruction-following or sentiment continuation task
# Expand the prompts_data with more diverse examples
prompts_data = {
    "query": [  # Changed from "prompt" to "query"
        "Generate a positive review for a movie about a space adventure:",
        "Write a short, coherent story about a friendly robot:",
        "Describe a delicious meal in a concise way:",
        "Continue this sentence with a factual statement: The capital of France is",
        "Explain the concept of gravity simply:",
        "Write a helpful response about learning programming:",
        "Describe the benefits of renewable energy:",
        "Explain why exercise is important for health:",
        "Write a creative story about time travel:",
        "Describe how to make a simple sandwich:",
        "Explain the water cycle in simple terms:",
        "Write about the importance of friendship:",
        "Describe a beautiful sunset:",
        "Explain how plants grow:",
        "Write about your favorite hobby:",
        "Describe the process of making coffee:"
    ] * 4  # Multiply to get enough samples
}

# Ensure we have at least BATCH_SIZE samples for training
if len(prompts_data["query"]) < BATCH_SIZE:
    prompts_data["query"] = prompts_data["query"] * ((BATCH_SIZE // len(prompts_data["query"])) + 1)

# Trim to exact batch size
prompts_data["query"] = prompts_data["query"][:BATCH_SIZE]

print(f"Dataset size: {len(prompts_data['query'])} samples")
dataset = Dataset.from_dict(prompts_data)

# Remove the tokenize_fn function and tokenization step
# The PPOTrainer will handle tokenization internally
def tokenize_fn(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=128) # Max prompt length

# --- 2. Initialize Model, Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # Set pad token for GPT-2

# PPOConfig sets up the PPO training parameters
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=LEARNING_RATE,
    ppo_epochs=PPO_EPOCHS,
    mini_batch_size=MINI_BATCH_SIZE,
    batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    log_with=None, # "wandb" or None
    tracker_project_name="trl_rllama_example",
    optimize_cuda_cache=True,
)

# Load model for PPO training
# TRL's AutoModelForCausalLMWithValueHead adds a value head to the model
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    ppo_config.model_name,
    # load_in_8bit=True, # If you have bitsandbytes and want to save memory
    # device_map="auto", # For multi-GPU or 8-bit loading
    peft_config=None, # Optional: PEFT config for LoRA, etc.
)
# Ensure model and tokenizer use the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- 3. Initialize RLlama Reward Processor ---
# This processor will use the rllama_config_trl.yaml to calculate rewards
try:
    rllama_processor = TRLRllamaRewardProcessor(
        rllama_config_path=RLLAMA_CONFIG_PATH,
        auto_register_llm_components=True
    )
    print(f"RLlama processor initialized with config: {RLLAMA_CONFIG_PATH}")
except Exception as e:
    print(f"Error initializing RLlama processor: {e}")
    print("Please ensure that the LLM reward components are properly implemented.")
    sys.exit(1)


# --- 4. Initialize PPOTrainer ---
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None, # If None, a copy of the model is used as the reference model
    tokenizer=tokenizer,
    dataset=dataset, # Can provide a dataset for batching, or handle batching manually
    data_collator=None, # Not strictly needed if we pass tokenized queries directly
)
print("PPOTrainer initialized.")

# --- 5. Training Loop ---
generation_kwargs = {
    "min_length": -1, # don't ignore the EOS token
    "top_k": 0.0,     # no top-k sampling
    "top_p": 1.0,     # no nucleus sampling
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": MAX_GEN_LENGTH,
    "eos_token_id": tokenizer.eos_token_id,
}

print(f"\nStarting TRL training with RLlama rewards for {NUM_TRAIN_STEPS} steps...")

for step in tqdm(range(NUM_TRAIN_STEPS), desc="PPO Training Steps"):
    # Get a batch of queries (prompts)
    try:
        # Sample a batch of queries
        sample_indices = torch.randperm(len(dataset))[:BATCH_SIZE].tolist()
        query_texts = [dataset[i]["query"] for i in sample_indices]
        query_tensors = []
        for query in query_texts:
            query_tensor = tokenizer.encode(query, return_tensors="pt").to(device)
            query_tensors.append(query_tensor.squeeze())
    except Exception as e:
        print(f"Error preparing queries: {e}")
        continue

    # Generate responses from the model
    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    
    # Decode responses to text
    response_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # --- 4. Compute rewards using RLlama ---
    try:
        # Use the correct method name
        rewards = rllama_processor.compute_rewards(
            prompts_text=query_texts,
            responses_text=response_texts,
            model_specific_infos=None
        )
        scores = [torch.tensor(float(r), dtype=torch.float32, device=device) for r in rewards]
    except Exception as e:
        print(f"Error computing RLlama rewards: {e}")
        rewards = [0.0 for _ in query_texts]
        scores = [torch.tensor(0.0, dtype=torch.float32, device=device) for _ in rewards]

    # --- 5. Perform PPO step ---
    try:
        stats = ppo_trainer.step(query_tensors, response_tensors, scores)
        
        if step % 1 == 0:
            print(f"\n--- Step {step+1}/{NUM_TRAIN_STEPS} ---")
            print(f"  Mean PPO reward: {stats.get('ppo/mean_scores', 'N/A')}")
            print(f"  Objective/kl: {stats.get('objective/kl', 'N/A')}")
            print(f"  Example Query: {query_texts[0][:100]}...")
            print(f"  Example Response: {response_texts[0][:100]}...")
            print(f"  Example Reward: {rewards[0]:.3f}")
            
    except Exception as e:
        print(f"Error during PPO step: {e}")
        continue

    # Reset RLlama components (e.g., normalizer stats) periodically if needed
    if (step + 1) % 50 == 0: # Example: reset every 50 steps
        rllama_processor.reset_components()

print("\nTraining finished.")

# --- Optional: Save the model ---
# output_dir = "./trl_rllama_finetuned_model"
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"Model saved to {output_dir}")

# --- Optional: Test generation with the fine-tuned model ---
# print("\nGenerating text with fine-tuned model:")
# test_prompt = "Generate a positive review for a new sci-fi movie:"
# inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
# gen_tokens = model.generate(**inputs, max_new_tokens=60, **generation_kwargs)
# generated_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
# print(f"Prompt: {test_prompt}")
# print(f"Generated: {generated_text}")

