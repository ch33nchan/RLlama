import re
import os
from tqdm import trange
import json
from datetime import datetime
import textworld.gym
from rllama import RLlamaAgent
import random
import argparse

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import numpy as np
from rllama.memory import MemoryEntry, EpisodicMemory, WorkingMemory, MemoryCompressor
import time

# Configure device for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps":
    torch.backends.mps.enable = True
else:
    device = "cpu"
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.enabled = False

class TextWorldAgent(RLlamaAgent):
    def __init__(self, model, tokenizer, device, generate_kwargs, ppo_kwargs, algorithm='ppo'):
        super().__init__(model, tokenizer, device, generate_kwargs, ppo_kwargs)
        self.algorithm = algorithm
        self.device = device
        self.setup_algorithm()
        self.saved_states = []
        self.saved_actions = []
        self.saved_dones = []
        self.current_episode_messages = []
        self.current_episode_rewards = []

    def _act_ppo(self, observation):
        prompt = self.format_observation(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(torch.int32).to(self.device)
        }
        
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": 100,
                "num_return_sequences": 1,
                "do_sample": True,
                "top_k": 20,
                "top_p": 0.9,
                "temperature": 0.7,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            outputs = self.model.generate(**generate_kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            state_input = self.get_state_embedding(observation)
            logits = self.policy_net(state_input)
            action_probs = F.softmax(logits, dim=-1)
            value = self.value_net(state_input)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            
            self.saved_states.append(state_input)
            self.saved_log_probs.append(dist.log_prob(action))
            self.saved_values.append(value)
            self.saved_actions.append(action)
            self.saved_dones.append(False)
            
            action_text = self.extract_action(response)
            return action_text if action_text is not None else "look"

    def llm(self, messages):
        # Ensure each message is a string
        prompt = "\n".join(str(message) if isinstance(message, dict) else message for message in messages)
        
        # Calculate max input length based on model's configuration
        max_input_length = min(2048, getattr(self.model.config, 'max_position_embeddings', 2048) - 100)
        
        # Tokenize with proper length constraints
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_input_length
        )
        
        # Convert tensors to the correct type and device
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(torch.int32).to(self.device)
        }
        
        with torch.no_grad():
            # Move computation to CPU if needed for certain operations
            if self.device == "mps":
                self.model = self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=100,
                    num_return_sequences=1,
                    do_sample=True,
                    top_k=20,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Move back to MPS if needed
                if self.device == "mps":
                    self.model = self.model.to(self.device)
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            except Exception as e:
                print(f"Generation error: {str(e)}")
                response = "command: look"
        
        return response

    def get_state_embedding(self, observation):
        obs_str = self.format_observation(observation)
        inputs = self.tokenizer(obs_str, return_tensors="pt")
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(torch.float32).to(self.device)
        }
        
        with torch.no_grad():
            outputs = self.model.pretrained_model(**inputs)
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
            return hidden_states.mean(dim=1)

    def setup_algorithm(self):
        self.gamma = 0.99
        self.learning_rate = 3e-4
        
        if self.algorithm == 'dqn':
            self.setup_dqn()
        elif self.algorithm == 'a2c':
            self.setup_a2c()
        elif self.algorithm == 'ppo':
            self.setup_ppo()
        elif self.algorithm == 'sac':
            self.setup_sac()
        elif self.algorithm == 'reinforce':
            self.setup_reinforce()
        elif self.algorithm == 'grpo':
            self.setup_grpo()

    def setup_dqn(self):
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.q_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def setup_a2c(self):
        self.value_net = nn.Linear(768, 1).to(self.device)
        self.policy_net = nn.Linear(768, 2).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=self.learning_rate)
        self.saved_actions = []
        self.saved_values = []

    def setup_ppo(self):
        self.clip_param = 0.2
        self.ppo_epochs = 4
        self.value_net = nn.Linear(768, 1).to(self.device)
        self.policy_net = nn.Linear(768, 2).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy_net.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=self.learning_rate)
        self.saved_log_probs = []
        self.saved_values = []

    def setup_sac(self):
        self.alpha = 0.2
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.q1_net = self.create_network()
        self.q2_net = self.create_network()
        self.target_q1_net = self.create_network()
        self.target_q2_net = self.create_network()
        self.policy_net = self.create_network()
        
        self.q_optimizer = torch.optim.Adam(
            list(self.q1_net.parameters()) + list(self.q2_net.parameters()),
            lr=self.learning_rate
        )
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

    def setup_reinforce(self):
        self.policy_net = nn.Linear(768, 2).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.saved_log_probs = []
        self.saved_rewards = []

    def setup_grpo(self):
        self.group_size = 5
        self.max_grad_norm = 1.0
        self.policy_net = nn.Linear(768, 2).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-5)
        self.current_group_episodes = []
        self.current_episode_log_probs = []

    def create_network(self):
        return nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        ).to(self.device)

    def get_system_prompt(self) -> str:
        return "You will be playing a text-based game. Here are some example commands: 'go west', 'inventory', 'drop teacup', 'examine broom', 'close type 2 box', 'open door', 'insert pencil into type 1 box', 'look'. Not all commands will work, but there are many others that you can try beyond these examples. When responding, first reason about the game state to decide the best action and then say 'command: <your command>'."

    def format_observation(self, observation) -> str:
        observation = observation.split("$$$$$$$ \n\n")[-1].strip()
        return observation

    def extract_action(self, response: str) -> str:
        command_match = re.search(r"(C|c)ommand: (.+?)(?=\n|$)", response)
        command = command_match.group(2) if command_match else None
        return command if command is not None else "look"

    def terminate_episode(self, train=True):
        if not train:
            return {}

        if self.algorithm == 'dqn':
            return self._train_dqn()
        elif self.algorithm == 'a2c':
            return self._train_a2c()
        elif self.algorithm == 'ppo':
            return self._train_ppo()
        elif self.algorithm == 'sac':
            return self._train_sac()
        elif self.algorithm == 'reinforce':
            return self._train_reinforce()
        elif self.algorithm == 'grpo':
            return self._train_grpo()

    def _train_dqn(self):
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def _train_a2c(self):
        if not self.saved_actions:
            return {"loss": 0.0}

        R = 0
        returns = []
        for r in reversed(self.current_episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)

        policy_losses = []
        value_losses = []
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_actions = []
        self.saved_values = []
        self.saved_log_probs = []
        
        return {"loss": loss.item()}

    def _train_ppo(self):
        if not self.saved_states:
            return {"loss": 0.0}

        states = torch.stack(self.saved_states)
        actions = torch.stack(self.saved_actions)
        old_probs = torch.stack(self.saved_log_probs)
        returns = torch.tensor(self._compute_returns()).to(self.device)
        advantages = returns - torch.cat(self.saved_values)

        total_loss = 0
        for _ in range(self.ppo_epochs):
            probs = self.policy_net(states)
            dist = Categorical(probs)
            new_probs = dist.log_prob(actions)
            
            ratio = (new_probs - old_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.smooth_l1_loss(self.value_net(states).squeeze(), returns)
            
            loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

        self.saved_states = []
        self.saved_actions = []
        self.saved_log_probs = []
        self.saved_values = []
        
        return {"loss": total_loss / self.ppo_epochs}

    def _train_sac(self):
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_states)
            next_q1 = self.target_q1_net(next_states, next_actions)
            next_q2 = self.target_q2_net(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q1 = self.q1_net(states, actions)
        current_q2 = self.q2_net(states, actions)
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        pi, log_pi = self._sample_action(states)
        q1_pi = self.q1_net(states, pi)
        q2_pi = self.q2_net(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}

    def _train_reinforce(self):
        R = 0
        returns = []
        for r in reversed(self.current_episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        
        return {"loss": policy_loss.item()}

    def _train_grpo(self):
        self.current_group_episodes.append({
            'log_probs': self.current_episode_log_probs,
            'rewards': self.current_episode_rewards
        })

        if len(self.current_group_episodes) >= self.group_size:
            total_loss = 0
            for episode in self.current_group_episodes:
                R = 0
                returns = []
                for r in reversed(episode['rewards']):
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns).to(self.device)

                policy_loss = []
                for log_prob, R in zip(episode['log_probs'], returns):
                    policy_loss.append(-log_prob * R)
                
                episode_loss = torch.stack(policy_loss).sum()
                total_loss += episode_loss

            loss = total_loss / len(self.current_group_episodes)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.current_group_episodes = []
            return {"loss": loss.item()}
        
        return {"loss": 0.0}

    def _compute_returns(self):
        R = 0
        returns = []
        for r in reversed(self.current_episode_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def _sample_action(self, state):
        mean, log_std = self.policy_net(state).chunk(2, dim=-1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def act(self, observation):
        state_embedding = self.get_state_embedding(observation)
        
        # Retrieve relevant past experiences
        relevant_memories = self.episodic_memory.retrieve_relevant(state_embedding, k=3)
        
        # Add to working memory
        for memory in relevant_memories:
            self.working_memory.add(memory.state)
        
        # Get context-enhanced state
        context = self.working_memory.get_context(state_embedding)
        
        # Use enhanced state for decision making
        action = super().act(observation)
        
        # Store experience
        self.episodic_memory.add(MemoryEntry(
            state=state_embedding,
            action=action,
            reward=None,
            next_state=None,
            done=False,
            timestamp=int(time.time())
        ))
        
        return action

    def assign_reward(self, reward):
        if self.episodic_memory.memories:
            latest_memory = self.episodic_memory.memories[-1]
            latest_memory.reward = reward
        super().assign_reward(reward)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TextWorld with different RL algorithms')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn', 'a2c', 'sac', 'reinforce', 'grpo'])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()
    # Update log directory path
    log_dir = f"/Users/cheencheen/Desktop/rl/RLlama/examples/logs/textworld_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    hyperparams = {
        "model_name": "facebook/opt-125m",
        "lora/r": 4,
        "lora/lora_alpha": 8,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": False,
        "batch_size": 1,
        "seed": 42069,
        "episodes": 20,
        "generate/max_new_tokens": 24,
        "generate/do_sample": True,
        "generate/top_p": 0.9,
        "generate/top_k": 20,
        "generate/temperature": 0.7,
    }
    
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(hyperparams, f, indent=2)
    
    HF_TOKEN = ""
    
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
    )
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        hyperparams["model_name"], 
        token=HF_TOKEN,
        padding_side="left",
        truncation_side="left",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
   
    special_tokens = {
        "pad_token": "<pad>",
        "sep_token": "<sep>",
        "bos_token": "<s>",
        "eos_token": "</s>"
    }
    tokenizer.add_special_tokens(special_tokens)
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
        algorithm=args.algorithm
    )

    env_id = textworld.gym.register_game(
        "/Users/cheencheen/Desktop/rl/RLlama/examples/tw_games/custom_game.z8",  # Updated path
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
        with open(f"{log_dir}/episode_{episode}.json", "w") as f:
            json.dump(episode_stats, f, indent=2)

    env.close()