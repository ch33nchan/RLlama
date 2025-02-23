import os
from tqdm import trange
import json
from datetime import datetime
import warnings
import torch
import random
import copy
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import re
import gymnasium as gym
from llamagym import Agent

import argparse

# Rest of the code remains the same...

class BlackjackAgent(Agent):
    def __init__(self, model, tokenizer, device, generate_kwargs, ppo_kwargs, algorithm='ppo'):
        super().__init__(model, tokenizer, device, generate_kwargs, ppo_kwargs)
        self.algorithm = algorithm
        self.setup_algorithm()

    def get_state_embedding(self, observation):
        # Convert observation to string format
        obs_str = self.format_observation(observation)
        
        # Tokenize the observation and ensure it's on the correct device with correct dtype
        inputs = self.tokenizer(obs_str, return_tensors="pt")
        inputs = {k: v.to(self.device).long() for k, v in inputs.items()}  # Ensure Long dtype
        
        # Get embeddings from the model
        with torch.no_grad():
            outputs = self.model.pretrained_model(**inputs)
            # Use hidden states from the last layer
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
            # Take mean across sequence length
            state_embedding = hidden_states.mean(dim=1)
        
        return state_embedding.to(self.device)

    def setup_algorithm(self):
        # Initialize common attributes
        self.current_group_episodes = []
        self.group_size = 5  # Default group size
        
        if self.algorithm == 'ppo':
            # PPO specific parameters
            self.clip_param = 0.2
            self.ppo_epochs = 4
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.value_loss_coef = 0.5
            self.entropy_coef = 0.01
            self.max_grad_norm = 0.5
            
            # Initialize value head
            self.value_net = nn.Linear(768, 1).to(self.device)
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.value_net.parameters()}
            ], lr=3e-4)
            
            # Storage for PPO
            self.saved_log_probs = []
            self.saved_values = []
            self.saved_states = []
            self.saved_actions = []
            self.saved_rewards = []
            self.saved_dones = []
            
        elif self.algorithm == 'dqn':
            # DQN specific parameters
            self.memory = deque(maxlen=10000)
            self.epsilon = 1.0
            self.epsilon_min = 0.01
            self.epsilon_decay = 0.995
            self.gamma = 0.99
            
            # Initialize networks
            self.target_model = copy.deepcopy(self.model)
            self.target_update_freq = 100
            self.step_counter = 0
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
            
        elif self.algorithm == 'a2c':
            # A2C specific parameters
            self.gamma = 0.99
            self.value_loss_coef = 0.5
            self.entropy_coef = 0.01
            self.max_grad_norm = 0.5
            
            # Initialize networks
            self.value_net = nn.Linear(768, 1).to(self.device)
            self.policy_net = nn.Linear(768, 2).to(self.device)  # 2 actions for Blackjack
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam([
                {'params': self.policy_net.parameters()},
                {'params': self.value_net.parameters()}
            ], lr=3e-4)
            
            # Storage
            self.saved_states = []
            self.saved_actions = []
            self.saved_log_probs = []
            self.saved_values = []
            self.saved_rewards = []
            self.saved_dones = []
            
        elif self.algorithm == 'reinforce':
            # REINFORCE specific parameters
            self.gamma = 0.99
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
            
            # Storage
            self.saved_log_probs = []
            self.saved_rewards = []
            self.current_episode_rewards = []
            
        elif self.algorithm == 'sac':
            # SAC specific parameters
            self.gamma = 0.99
            self.alpha = 0.2  # Temperature parameter
            self.target_entropy = -1.0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            
            # Initialize networks
            self.q1_net = nn.Linear(768, 2).to(self.device)
            self.q2_net = nn.Linear(768, 2).to(self.device)
            self.target_q1_net = copy.deepcopy(self.q1_net)
            self.target_q2_net = copy.deepcopy(self.q2_net)
            
            # Initialize optimizers
            self.q_optimizer = torch.optim.Adam([
                {'params': self.q1_net.parameters()},
                {'params': self.q2_net.parameters()}
            ], lr=3e-4)
            self.policy_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
            
            # Storage
            self.memory = deque(maxlen=10000)

    def act(self, observation):
        state_embedding = self.get_state_embedding(observation)
        
        if self.algorithm == 'reinforce':
            with torch.no_grad():
                # Convert state embedding to float and ensure correct shape
                state_input = state_embedding.float()
                logits = self.model.v_head(state_input)
                action_probs = F.softmax(logits, dim=-1)
                
                # Sample action using categorical distribution
                dist = Categorical(action_probs)
                action = dist.sample()
                
                # Store log probability for training
                self.saved_log_probs.append(dist.log_prob(action))
            
            return action.item()
            
        elif self.algorithm == 'a2c':
            # A2C implementation
            with torch.no_grad():
                policy_logits = self.policy_net(state_embedding)
                action_probs = F.softmax(policy_logits, dim=-1)
                value = self.value_net(state_embedding)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                
                self.saved_states.append(state_embedding)
                self.saved_log_probs.append(dist.log_prob(action))
                self.saved_values.append(value)
                self.saved_actions.append(action)
                self.saved_dones.append(False)
                
            return action.item()
            
        elif self.algorithm == 'ppo':
            # PPO implementation
            with torch.no_grad():
                state_input = state_embedding.float()
                logits = self.model.v_head(state_input)
                action_probs = F.softmax(logits, dim=-1)
                value = self.value_net(state_input)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                
                self.saved_states.append(state_embedding)
                self.saved_log_probs.append(dist.log_prob(action))
                self.saved_values.append(value)
                self.saved_actions.append(action)
                self.saved_dones.append(False)
                
            return action.item()
            
        elif self.algorithm == 'dqn':
            if random.random() < self.epsilon:
                return random.randint(0, 1)
            with torch.no_grad():
                q_values = self.model(state_embedding)
                return torch.argmax(q_values).item()
                
        elif self.algorithm == 'reinforce':
            action_probs = F.softmax(self.model(state_embedding), dim=-1)
            m = Categorical(action_probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            return action.item()
            
        elif self.algorithm == 'sac':
            mean, log_std = self.model(state_embedding).chunk(2, dim=-1)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            return action.item()

    def update(self, batch):
        if self.algorithm == 'ppo':
            states = torch.stack(self.saved_states)
            actions = torch.tensor(self.saved_actions, device=self.device)
            old_log_probs = torch.stack(self.saved_log_probs)
            old_values = torch.stack(self.saved_values)
            rewards = torch.tensor(self.saved_rewards, device=self.device)
            dones = torch.tensor(self.saved_dones, device=self.device)
            
            # Compute returns and advantages
            returns = self.compute_returns(rewards, dones)
            advantages = returns - old_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(self.ppo_epochs):
                # Get current policy and value predictions
                action_probs = F.softmax(self.model(states), dim=-1)
                current_values = self.value_net(states).squeeze()
                
                # Compute ratio and clipped ratio
                ratio = torch.exp(torch.log(action_probs.gather(1, actions.unsqueeze(1))) - old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                
                # Compute losses
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                value_loss = F.mse_loss(current_values, returns)
                entropy_loss = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))
                
                # Combined loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Clear storage
            self.saved_log_probs = []
            self.saved_values = []
            self.saved_states = []
            self.saved_actions = []
            self.saved_rewards = []
            self.saved_dones = []
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy_loss.item()
            }
            
        elif self.algorithm == 'dqn':
            states, actions, rewards, next_states, dones = batch
            current_q = self.model(states).gather(1, actions)
            next_q = self.target_model(next_states).max(1)[0].detach()
            target_q = rewards + self.gamma * next_q * (1 - dones)
            loss = F.mse_loss(current_q, target_q.unsqueeze(1))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.step_counter += 1
            if self.step_counter % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
        elif self.algorithm == 'a2c':
            states, actions, rewards, values = batch
            returns = self.compute_returns(rewards)
            advantages = returns - values
            
            action_probs = F.softmax(self.model(states), dim=-1)
            log_probs = torch.log(action_probs.gather(1, actions))
            
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = F.mse_loss(values, returns)
            
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        elif self.algorithm == 'reinforce':
            returns = self.compute_returns(self.current_episode_rewards)
            policy_loss = []
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
        elif self.algorithm == 'sac':
            states, actions, rewards, next_states, dones = batch
            
            # Update Q-functions
            current_q1 = self.q1_net(states)
            current_q2 = self.q2_net(states)
            next_value = self.compute_next_value(next_states)
            target_q = rewards + self.gamma * next_value * (1 - dones)
            
            q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Update policy
            new_actions, log_probs = self.compute_actions_and_probs(states)
            min_q = torch.min(self.q1_net(states), self.q2_net(states))
            policy_loss = (self.alpha * log_probs - min_q).mean()
            
            # Update alpha
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

    def compute_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        last_return = 0
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = self.saved_values[t + 1].item()
                
            returns[t] = rewards[t] + self.gamma * next_value * (1 - dones[t])
            td_error = rewards[t] + self.gamma * next_value * (1 - dones[t]) - self.saved_values[t].item()
            advantages[t] = td_error + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
            
        return returns

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
            if len(episode['log_probs']) > 0:  # Only add if there are log probs
                all_messages.extend(episode['messages'])
                all_log_probs.extend(episode['log_probs'])
                flattened_rewards.extend([reward] * len(episode['log_probs']))
        
        # Check if we have any data to process
        if not all_log_probs:
            return {
                'policy_loss': 0.0,
                'mean_reward': sum(relative_rewards) / len(relative_rewards) if relative_rewards else 0.0,
                'std_reward': 0.0
            }
        
        # Convert to tensors with gradients enabled
        rewards_tensor = torch.tensor(flattened_rewards, device=self.device, requires_grad=False)
        log_probs_tensor = torch.stack(all_log_probs)  # Use stack instead of tensor
        
        # Compute policy loss with detached rewards
        policy_loss = -(log_probs_tensor * rewards_tensor.detach()).mean()
        
        # Update model
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'mean_reward': rewards_tensor.mean().item(),
            'std_reward': rewards_tensor.std().item()
        }

# Add this after the imports
def check_mps_availability():
    print("\nChecking MPS availability:")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    try:
        # Try to create MPS device
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print("MPS device test successful")
        return True
    except Exception as e:
        print(f"MPS device test failed: {e}")
        return False

# Modify the device setup (replace the existing device setup)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'dqn', 'a2c', 'reinforce', 'sac', 'grpo'],
                       default='ppo',
                       help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    args = parser.parse_args()

    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create log directory
    log_dir = f"logs/blackjack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    hyperparams = {
        "model_name": "facebook/opt-125m",  # Much smaller model
        "env": "Blackjack-v1",
        "lora/r": 2,                 # Reduced LoRA rank further
        "lora/lora_alpha": 4,        # Reduced alpha
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": False,
        "batch_size": 1,             
        "seed": 42069,
        "episodes": 10,              # Reduced episodes further
        "generate/max_new_tokens": 16,  # Reduced tokens
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
    
    # Replace the device setup section in __main__
    # Device setup
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Set default device
    torch.set_default_device(device)
    
    # Create model with explicit device placement
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN,
        device_map={"": device},  # Explicit device mapping
        output_hidden_states=True  # Add this line
    )
    
    # Ensure model and its components are on the correct device
    model = model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    
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
        # Save episode stats
        with open(f"{log_dir}/episode_{episode}.json", "w") as f:
            json.dump(episode_stats, f, indent=2)

        # Save model checkpoint every 5 episodes
        if (episode + 1) % 5 == 0:
            checkpoint_path = f"{log_dir}/checkpoint_{episode+1}"
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_episodes = 10
    eval_rewards = []
    
    for eval_ep in range(eval_episodes):
        observation, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            with torch.no_grad():
                action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        eval_rewards.append(ep_reward)

    # Save final statistics
    final_stats = {
        "training_episodes": hyperparams["episodes"],
        "average_training_reward": sum(agent.current_episode_rewards) / hyperparams["episodes"],
        "eval_episodes": eval_episodes,
        "average_eval_reward": sum(eval_rewards) / eval_episodes,
        "algorithm": args.algorithm,
        "final_epsilon": agent.epsilon if hasattr(agent, "epsilon") else None,
    }

    with open(f"{log_dir}/final_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    print("\nFinal Statistics:")
    print(f"Average training reward: {final_stats['average_training_reward']:.2f}")
    print(f"Average evaluation reward: {final_stats['average_eval_reward']:.2f}")

    # Save final model
    final_model_path = f"{log_dir}/final_model"
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    env.close()