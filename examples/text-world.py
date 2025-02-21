import re
import os
from tqdm import trange
import json
from datetime import datetime
import textworld.gym
from llamagym import Agent

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead


# Add torch import at the top
import torch

class TextworldAgent(Agent):
    def __init__(self, model, tokenizer, device, generate_kwargs, ppo_kwargs):
        super().__init__(model, tokenizer, device, generate_kwargs, ppo_kwargs)

    def get_system_prompt(self) -> str:
        return "You will be playing a text-based game. Here are some example commands: 'go west', 'inventory', 'drop teacup', 'examine broom', 'close type 2 box', 'open door', 'insert pencil into type 1 box', 'look'. Not all commands will work, but there are many others that you can try beyond these examples. When responding, first reason about the game state to decide the best action and then say 'command: <your command>'."

    def format_observation(self, observation) -> str:
        # remove the game header text
        observation = observation.split("$$$$$$$ \n\n")[-1].strip()
        return observation

    def extract_action(self, response: str) -> str:
        command_match = re.search(r"(C|c)ommand: (.+?)(?=\n|$)", response)
        command = command_match.group(2) if command_match else None
        return command if command is not None else "look"

    def act(self, observation):
        prompt = self.format_observation(observation)
        # Ensure consistent device usage
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generate_kwargs = {
            "max_new_tokens": 100,
            "num_return_sequences": 1,
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.7,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask", None),
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        outputs = self.model.generate(**generate_kwargs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        action = self.extract_action(response)
        
        return action if action is not None else "look"

if __name__ == "__main__":
    # Create log directory
    log_dir = f"logs/textworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    hyperparams = {
        "model_name": "distilgpt2",  # Much smaller model
        "lora/r": 4,                 # Reduced LoRA rank
        "lora/lora_alpha": 8,        # Reduced alpha
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": False,
        "batch_size": 1,             # Minimum batch size
        "seed": 42069,
        "episodes": 20,              # Reduced episodes
        "generate/max_new_tokens": 24,# Reduced token generation
        "generate/do_sample": True,
        "generate/top_p": 0.9,
        "generate/top_k": 20,
        "generate/temperature": 0.7,
    }
    ß
    # Save hyperparams
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(hyperparams, f, indent=2)
    
    # Force CPU usage and disable MPS/CUDA
    import torch
    torch.set_default_device('cpu')
    device = "cpu"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    # Create LoRA config before model initialization
    lora_config = LoraConfig(
        **{
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("lora/")
        }
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN,
        device_map=None,  # Disable auto device mapping
    ).cpu()  # Explicitly move to CPU
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    agent = TextworldAgent(
        model,
        tokenizer,
        device,
        {
            key: value
            for key, value in hyperparams.items()
            if key.startswith("generate/")
        },
        {
            "batch_size": hyperparams["batch_size"],
            "mini_batch_size": hyperparams["batch_size"],
        },
    )

    env_id = textworld.gym.register_game(
        "examples/tw_games/custom_game.z8",
        max_episode_steps=50,
        request_infos=textworld.EnvInfos(
            admissible_commands=True,
        ),
    )
    env = textworld.gym.make(env_id)

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        env.render()
        done = False

        while not done:
            action = agent.act(observation)
            # Removed wandb logging
            observation, reward, done, info = env.step(action)
            env.render()
            agent.assign_reward(reward)

        episode_stats = {
            "episode": episode,
            "total_return": sum(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
            "episode_messages": agent.current_episode_messages[-1],
        }
        train_stats = agent.terminate_episode()
        episode_stats.update(train_stats)
        # Log to file instead of wandb
        with open(f"{log_dir}/episode_{episode}.json", "w") as f:
            json.dump(episode_stats, f, indent=2)

    env.close()
