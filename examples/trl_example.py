# examples/trl_example.py

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer

# Import our custom reward processor
from rllama.integration import TRLRlamaRewardProcessor

# 1. SETUP: Define configurations
# TRL training configuration
config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=16,
    log_with="tensorboard",
    project_kwargs={"logging_dir": "./logs/trl_example"},
    # We will use a smaller model for this example
    ppo_epochs=4,
    kl_penalty="kl",
    adap_kl_ctrl=True,
    target_kl=0.1,
)

# Generation configuration
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 50256, # EOS token ID for GPT-2
    "max_new_tokens": 32,
}

# 2. MODEL and TOKENIZER
# Load a pre-trained model and tokenizer
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. DATASET
# Load a dataset for training
dataset = load_dataset("imdb", split="train")
dataset = dataset.shuffle().select(range(1000)) # Use a small subset for the example

def tokenize(sample):
    # Truncate prompt to avoid excessive memory usage
    sample["input_ids"] = tokenizer.encode(sample["text"])[:32]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type="torch")

# 4. RLLAMA INTEGRATION
# This is the key step: instantiate our reward processor with the YAML config.
# It automatically loads all components, the composer, and the shaper.
try:
    reward_processor = TRLRlamaRewardProcessor(config_path="examples/reward_config.yaml")
except Exception as e:
    print(f"Error loading reward system: {e}")
    exit()

# 5. TRL TRAINER INITIALIZATION
# Pass the `compute_reward` method directly to the PPOTrainer.
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    compute_reward=reward_processor.compute_reward
)

# 6. TRAINING LOOP
print("Starting training...")
for epoch in tqdm(range(config.epochs), "epoch"):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # Get model response
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, batch["response"])
        ppo_trainer.log_stats(stats, batch, [torch.tensor(0.0)] * len(query_tensors))

print("✅ Training finished successfully!")