import os
from tqdm import trange
import json
from datetime import datetime
import warnings
import torch

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import re
import gymnasium as gym
from llamagym import Agent


import argparse  # Add at the top with other imports

class BlackjackAgent(Agent):
    def __init__(self, model, tokenizer, device, generate_kwargs, ppo_kwargs, algorithm='ppo'):
        super().__init__(model, tokenizer, device, generate_kwargs, ppo_kwargs)
        self.algorithm = algorithm
        if algorithm == 'grpo':
            self.group_buffer = []
            self.group_size = 8
            self.current_group_episodes = []
            self.current_episode_log_probs = []

    def get_system_prompt(self) -> str:
        return """You are an expert blackjack player. Follow these strict rules:
1. ALWAYS hit (Action: 1) if your total is 11 or below - no exceptions!
2. With 12-16:
   - Hit (Action: 1) if dealer shows 7 or higher
   - Stay (Action: 0) if dealer shows 6 or lower
3. ALWAYS stay (Action: 0) if your total is 17 or higher without an ace
4. With a usable ace:
   - Hit (Action: 1) if your total is 17 or below
   - Stay (Action: 0) if your total is 18 or above
Respond ONLY with 'Action: 1' to hit or 'Action: 0' to stay."""

    def format_observation(self, observation: gym.core.ObsType) -> str:
        return f"Current hand total: {observation[0]}\nDealer's card: {observation[1]}\nUsable ace: {'yes' if bool(observation[2]) else 'no'}\nWhat is your action? Respond ONLY with Action: 0 or Action: 1."

    def extract_action(self, response: str) -> gym.core.ActType:
        match = re.compile(r"Action: (\d)").search(response)
        if match:
            return int(match.group(1))

        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1

        return 0

    def act(self, observation):
        prompt = self.format_observation(observation)
        print(f"\nCurrent state: {prompt}")  # Debug output
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generate_kwargs = {
            "max_new_tokens": 24,
            "num_return_sequences": 1,
            "do_sample": True,
            "top_k": 20,
            "top_p": 0.9,
            "temperature": 0.7,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask", None),
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": True
        }
        
        self.model = self.model.to(self.device)
        outputs = self.model.generate(**generate_kwargs)
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Store log probabilities for GRPO
        if self.algorithm == 'grpo':
            logits = outputs.scores[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            action_probs = probs[0][outputs.sequences[0][-1]]
            self.current_episode_log_probs.append(torch.log(action_probs).item())
        
        print(f"Model response: {response}")  # Debug output
        action = self.extract_action(response)
        print(f"Action taken: {'stay' if action == 0 else 'hit'}")  # Debug output
        
        return action

    def terminate_episode(self):
        if self.algorithm == 'ppo':
            return super().terminate_episode()
        else:  # GRPO
            if not hasattr(self, 'current_episode_log_probs'):
                self.current_episode_log_probs = []
                
            self.current_group_episodes.append({
                'rewards': self.current_episode_rewards.copy(),
                'messages': self.current_episode_messages.copy(),
                'log_probs': self.current_episode_log_probs.copy()
            })
            
            # Reset episode data
            self.current_episode_log_probs = []
            
            if len(self.current_group_episodes) >= self.group_size:
                group_returns = [sum(ep['rewards']) for ep in self.current_group_episodes]
                mean_return = sum(group_returns) / len(group_returns)
                relative_rewards = [ret - mean_return for ret in group_returns]
                
                stats = self._update_policy_grpo(relative_rewards)
                self.current_group_episodes = []
                return stats
            return None

    def _update_policy_grpo(self, relative_rewards):
        all_messages = []
        all_log_probs = []
        flattened_rewards = []
        
        # Flatten episodes and expand rewards
        for episode, reward in zip(self.current_group_episodes, relative_rewards):
            all_messages.extend(episode['messages'])
            all_log_probs.extend(episode['log_probs'])
            flattened_rewards.extend([reward] * len(episode['messages']))
        
        # Convert to tensors with gradients enabled
        rewards_tensor = torch.tensor(flattened_rewards, device=self.device, requires_grad=False)
        log_probs_tensor = torch.tensor(all_log_probs, device=self.device, requires_grad=True)
        
        # Compute policy loss with detached rewards
        policy_loss = -(log_probs_tensor * rewards_tensor.detach()).mean()
        
        # Update model
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Add gradient clipping
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'mean_reward': rewards_tensor.mean().item(),
            'std_reward': rewards_tensor.std().item()
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'grpo'], default='ppo',
                      help='RL algorithm to use (ppo or grpo)')
    args = parser.parse_args()

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create log directory
    log_dir = f"logs/blackjack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    hyperparams = {
        "model_name": "distilgpt2",  # Changed to smaller model
        "env": "Blackjack-v1",
        "lora/r": 4,                 # Reduced LoRA rank
        "lora/lora_alpha": 8,        # Reduced alpha
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": False,
        "batch_size": 1,             # Reduced batch size
        "seed": 42069,
        "episodes": 20,              # Reduced episodes
        "generate/max_new_tokens": 24,
        "generate/do_sample": True,
        "generate/top_p": 0.9,
        "generate/top_k": 20,
        "generate/temperature": 0.7,
    }
    
    # Save hyperparams
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(hyperparams, f, indent=2)
    
    # Force CPU usage
    torch.set_default_device('cpu')
    device = "cpu"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    # Create LoRA config
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

    agent = BlackjackAgent(
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
        algorithm=args.algorithm
    )
    env = gym.make(hyperparams["env"], natural=False, sab=False)

    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False
        total_reward = 0  # Track episode reward
        print(f"\nStarting Episode {episode + 1}")
        
        episode_actions = []
        while not done:
            action = agent.act(observation)
            episode_actions.append(action)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Reward from last action: {reward}")  # Print immediate reward
            agent.assign_reward(reward)
            done = terminated or truncated

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")
        
        episode_stats = {
            "episode": episode,
            "total_return": total_reward,  # Use tracked reward
            "message_ct": len(agent.current_episode_messages),
            "actions": episode_actions,
            "final_reward": reward  # Add final reward
        }
        train_stats = agent.terminate_episode()
        if train_stats:
            episode_stats.update(train_stats)
        
        # Log to file
        with open(f"{log_dir}/episode_{episode}.json", "w") as f:
            json.dump(episode_stats, f, indent=2)

    print("\nFinal Statistics:")
    print(f"Average reward per episode: {sum(agent.current_episode_rewards) / hyperparams['episodes']:.2f}")

    env.close()