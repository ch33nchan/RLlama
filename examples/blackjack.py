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
from rllama import RLlamaAgent  # Changed from llamagym import Agent

import argparse
from rllama.memory import MemoryEntry, EpisodicMemory, WorkingMemory, MemoryCompressor
import time

# Rest of the code remains the same...

class BlackjackAgent(RLlamaAgent):  # Changed from Agent to RLlamaAgent
    def __init__(self, model, tokenizer, device, generate_kwargs, ppo_kwargs, algorithm='ppo'):
        super().__init__(model, tokenizer, device, generate_kwargs, ppo_kwargs, algorithm)
        self.algorithm = algorithm
        # Add memory components
        self.episodic_memory = EpisodicMemory(capacity=1000)
        self.working_memory = WorkingMemory(max_size=5)
        self.memory_compressor = MemoryCompressor(compression_ratio=0.5)
        self.setup_algorithm()

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

    def get_state_embedding(self, observation):  # Fixed indentation
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
        
        # Fallback parsing
        digits = [char for char in response if char.isdigit()]
        if len(digits) == 0 or digits[-1] not in ("0", "1"):
            if "stick" in response.lower():
                return 0
            elif "hit" in response.lower():
                return 1
        
        return 0

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
            self.batch_size = 32  # Add batch size parameter
            
            # Initialize networks
            self.target_model = copy.deepcopy(self.model)
            self.target_update_freq = 100
            self.step_counter = 0
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
            
            # Add storage for current episode
            self.current_state = None
            self.current_action = None
            
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
            self.tau = 0.005  # Target network update rate
            self.batch_size = 32  # Add batch size parameter
            
            # Initialize networks
            self.q1_net = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            ).to(self.device)
            
            self.q2_net = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            ).to(self.device)
            
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
            self.saved_states = []
            self.saved_actions = []
            self.saved_log_probs = []
            self.saved_rewards = []
            self.saved_dones = []
            
        elif self.algorithm == 'grpo':
            # GRPO specific parameters
            self.gamma = 0.99
            self.group_size = 5
            self.max_grad_norm = 1.0
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
            
            # Storage
            self.current_group_episodes = []
            self.current_episode_log_probs = []
            self.current_episode_rewards = []
            self.current_episode_messages = []

    def act(self, observation):
        state_embedding = self.get_state_embedding(observation)
        
        if self.algorithm == 'grpo':
            # Convert state embedding to float and ensure correct shape
            state_input = state_embedding.float()
            logits = self.model.v_head(state_input)
            action_probs = F.softmax(logits, dim=-1)
            
            # Sample action using categorical distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # Store log probability with gradient
            log_prob = dist.log_prob(action)
            self.current_episode_log_probs.append(log_prob)
            
            # Store the action for later use
            action = action.clamp(0, 1)
            
            return action.item()
            
        elif self.algorithm == 'ppo':
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
                
                action = action.clamp(0, 1)
            
            return action.item()
            
        elif self.algorithm == 'dqn':
            with torch.no_grad():
                state_input = state_embedding.float()
                
                # Epsilon-greedy action selection
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)
                else:
                    q_values = self.model.v_head(state_input)
                    action = torch.argmax(q_values).item()
                
                # Store current action
                self.current_action = action
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
            return action
            
        elif self.algorithm == 'a2c':
            with torch.no_grad():
                state_input = state_embedding.float()
                policy_logits = self.policy_net(state_input)
                value = self.value_net(state_input)
                
                action_probs = F.softmax(policy_logits, dim=-1)
                dist = Categorical(action_probs)
                action = dist.sample()
                
                self.saved_states.append(state_embedding)
                self.saved_log_probs.append(dist.log_prob(action))
                self.saved_values.append(value)
                self.saved_actions.append(action)
                self.saved_dones.append(False)
                
                action = action.clamp(0, 1)
            
            return action.item()
            
        elif self.algorithm == 'reinforce':
            with torch.no_grad():
                state_input = state_embedding.float()
                logits = self.model.v_head(state_input)
                action_probs = F.softmax(logits, dim=-1)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                
                self.saved_log_probs.append(dist.log_prob(action))
                action = action.clamp(0, 1)
            
            return action.item()
            
        elif self.algorithm == 'sac':
            with torch.no_grad():
                state_input = state_embedding.float()
                logits = self.model.v_head(state_input)
                action_probs = F.softmax(logits, dim=-1)
                
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                self.saved_states.append(state_embedding)
                self.saved_actions.append(action)
                self.saved_log_probs.append(log_prob)
                
                action = action.clamp(0, 1)
            
            return action.item()

    def compute_returns_and_advantages(self):
        # Check if we have any rewards
        if not self.saved_rewards:
            return [], []
            
        returns = []
        advantages = []
        next_value = 0
        next_advantage = 0
        
        for reward, value, done in zip(reversed(self.saved_rewards), 
                                     reversed(self.saved_values), 
                                     reversed(self.saved_dones)):
            if done:
                next_return = 0
                next_advantage = 0
            
            next_return = reward + self.gamma * next_value * (1 - done)
            next_advantage = reward + self.gamma * next_value * (1 - done) - value.item()
            
            returns.insert(0, next_return)
            advantages.insert(0, next_advantage)
            
            next_value = value.item()
            
        return returns, advantages

    def terminate_episode(self):
        if self.algorithm == 'ppo':
            # Check if we have any data to process
            if not self.saved_states or not self.saved_actions:
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0
                }
                
            returns, advantages = self.compute_returns_and_advantages()
            if not returns:  # If no returns were computed
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0
                }
            
            # Rest of the PPO update code remains the same
            old_states = torch.stack(self.saved_states)
            old_actions = torch.stack(self.saved_actions)
            old_log_probs = torch.stack(self.saved_log_probs)
            
            for _ in range(self.ppo_epochs):
                # Get current policy and value
                logits = self.model.v_head(old_states.float())
                action_probs = F.softmax(logits, dim=-1)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(old_actions)
                values = self.value_net(old_states.float())
                
                # Calculate ratios and surrogate losses
                ratios = torch.exp(new_log_probs - old_log_probs)
                advantages_tensor = torch.tensor(advantages, device=self.device)
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages_tensor
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), torch.tensor(returns, device=self.device))
                entropy = dist.entropy().mean()
                
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update model
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # Clear episode data
            self.saved_states = []
            self.saved_actions = []
            self.saved_log_probs = []
            self.saved_values = []
            self.saved_rewards = []
            self.saved_dones = []
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item()
            }
            
        elif self.algorithm == 'dqn':
            # DQN update
            if len(self.memory) < self.batch_size:
                return None
                
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states).float()
            next_states = torch.stack(next_states).float()
            actions = torch.tensor(actions, device=self.device)
            rewards = torch.tensor(rewards, device=self.device)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            # Get current Q values
            current_q_values = self.model.v_head(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            # Get next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_model.v_head(next_states)
                next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss and update
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network
            self.step_counter += 1
            if self.step_counter % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            return {'loss': loss.item()}
            
        elif self.algorithm == 'a2c':
            # Check if we have any data to process
            if not self.saved_states or not self.saved_actions or not self.saved_rewards:
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0
                }
            
            returns, advantages = self.compute_returns_and_advantages()
            if not returns:  # If no returns were computed
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': 0.0
                }
            
            # Convert to tensors
            states = torch.stack(self.saved_states)
            actions = torch.stack(self.saved_actions)
            
            # Get current policy and value
            policy_logits = self.policy_net(states.float())
            values = self.value_net(states.float())
            
            # Calculate losses
            advantages_tensor = torch.tensor(advantages, device=self.device)
            returns_tensor = torch.tensor(returns, device=self.device)
            
            dist = Categorical(F.softmax(policy_logits, dim=-1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            policy_loss = -(log_probs * advantages_tensor).mean()
            value_loss = F.mse_loss(values.squeeze(), returns_tensor)
            
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Update model
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Clear episode data
            self.saved_states = []
            self.saved_actions = []
            self.saved_log_probs = []
            self.saved_values = []
            self.saved_rewards = []
            self.saved_dones = []
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item()
            }
            
        elif self.algorithm == 'reinforce':
            # Check if we have any data to process
            if not self.saved_log_probs or not self.saved_rewards:
                return {'policy_loss': 0.0}
            
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(self.saved_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
                
            if not returns:  # If no returns were computed
                return {'policy_loss': 0.0}
                
            returns = torch.tensor(returns, device=self.device)
            
            # Normalize returns
            if len(returns) > 1:  # Only normalize if we have more than one return
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Calculate loss
            policy_loss = []
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
                
            if not policy_loss:  # If no policy loss was computed
                return {'policy_loss': 0.0}
                
            policy_loss = torch.stack(policy_loss).sum()
            
            # Update model
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # Clear episode data
            self.saved_log_probs = []
            self.saved_rewards = []
            
            return {'policy_loss': policy_loss.item()}
            
        elif self.algorithm == 'sac':
            if len(self.memory) < self.batch_size:
                return None
                
            # Sample from memory
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states).float()
            next_states = torch.stack(next_states).float()
            actions = torch.tensor(actions, device=self.device)
            rewards = torch.tensor(rewards, device=self.device)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            # Update Q networks
            with torch.no_grad():
                next_state_logits = self.model.v_head(next_states)
                next_state_probs = F.softmax(next_state_logits, dim=-1)
                next_state_dist = Categorical(next_state_probs)
                next_state_actions = next_state_dist.sample()
                next_state_log_probs = next_state_dist.log_prob(next_state_actions)
                
                next_q1 = self.target_q1_net(next_states)
                next_q2 = self.target_q2_net(next_states)
                next_q = torch.min(next_q1, next_q2)
                next_q = next_q.gather(1, next_state_actions.unsqueeze(1)).squeeze()
                
                target_q = rewards + (1 - dones) * self.gamma * (next_q - self.alpha * next_state_log_probs)
            
            current_q1 = self.q1_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            current_q2 = self.q2_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            q1_loss = F.mse_loss(current_q1, target_q)
            q2_loss = F.mse_loss(current_q2, target_q)
            
            # Update policy
            logits = self.model.v_head(states)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_actions = dist.sample()
            log_probs = dist.log_prob(new_actions)
            
            q1_new = self.q1_net(states).gather(1, new_actions.unsqueeze(1)).squeeze()
            q2_new = self.q2_net(states).gather(1, new_actions.unsqueeze(1)).squeeze()
            q_new = torch.min(q1_new, q2_new)
            
            policy_loss = (self.alpha * log_probs - q_new).mean()
            
            # Update temperature parameter
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            # Perform updates
            self.q_optimizer.zero_grad()
            (q1_loss + q2_loss).backward()
            self.q_optimizer.step()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update target networks
            for target_param, param in zip(self.target_q1_net.parameters(), self.q1_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_q2_net.parameters(), self.q2_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.alpha = self.log_alpha.exp()
            
            return {
                'q1_loss': q1_loss.item(),
                'q2_loss': q2_loss.item(),
                'policy_loss': policy_loss.item(),
                'alpha_loss': alpha_loss.item(),
                'alpha': self.alpha.item()
            }
            
        elif self.algorithm == 'grpo':
            # Check if we have any data to process
            if not self.current_episode_log_probs or not self.current_episode_rewards:
                return {
                    'policy_loss': 0.0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0
                }
            
            # Calculate episode return
            episode_return = sum(self.current_episode_rewards)
            
            # Store current episode data
            self.current_group_episodes.append({
                'log_probs': self.current_episode_log_probs.copy(),
                'rewards': [episode_return]  # Store total episode reward
            })
            
            # Only update policy when we have enough episodes
            if len(self.current_group_episodes) >= self.group_size:
                # Calculate relative rewards within the group
                episode_returns = [sum(ep['rewards']) for ep in self.current_group_episodes]
                mean_return = sum(episode_returns) / len(episode_returns)
                std_return = torch.std(torch.tensor(episode_returns, device=self.device)).item() if len(episode_returns) > 1 else 0
                
                # Normalize rewards
                if std_return > 0:
                    relative_rewards = [(ret - mean_return) / std_return for ret in episode_returns]
                else:
                    relative_rewards = [ret - mean_return for ret in episode_returns]
                
                # Calculate policy loss
                policy_loss = torch.tensor(0.0, device=self.device)
                for episode, reward in zip(self.current_group_episodes, relative_rewards):
                    if len(episode['log_probs']) > 0:
                        episode_loss = torch.stack([log_prob * (-reward) for log_prob in episode['log_probs']]).sum()
                        policy_loss = policy_loss + episode_loss
                
                # Update model if we have a non-zero loss
                if torch.abs(policy_loss) > 1e-8:
                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Clear group data
                self.current_group_episodes = []
                
                stats = {
                    'policy_loss': policy_loss.item(),
                    'mean_reward': mean_return,
                    'std_reward': std_return
                }
            else:
                stats = {
                    'policy_loss': 0.0,
                    'mean_reward': episode_return,
                    'std_reward': 0.0
                }
            
            # Clear episode data
            self.current_episode_log_probs = []
            self.current_episode_rewards = []
            
            return stats

    def _update_policy_grpo(self, relative_rewards):
        all_log_probs = []
        flattened_rewards = []
        
        # Flatten episodes and expand rewards
        for episode, reward in zip(self.current_group_episodes, relative_rewards):
            if len(episode['log_probs']) > 0:
                all_log_probs.extend(episode['log_probs'])
                flattened_rewards.extend([reward] * len(episode['log_probs']))
        
        if not all_log_probs:
            return {
                'policy_loss': 0.0,
                'mean_reward': sum(relative_rewards) / len(relative_rewards) if relative_rewards else 0.0,
                'std_reward': 0.0
            }
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(flattened_rewards, device=self.device)
        
        # Store current episode data
        self.current_group_episodes.append({
            'log_probs': self.current_episode_log_probs,
            'rewards': self.current_episode_rewards
        })
        
        # Only update policy when we have enough episodes
        if len(self.current_group_episodes) >= self.group_size:
            # Calculate relative rewards within the group
            episode_returns = [sum(ep['rewards']) for ep in self.current_group_episodes]
            if len(episode_returns) > 0:  # Prevent division by zero
                mean_return = sum(episode_returns) / len(episode_returns)
                relative_rewards = [ret - mean_return for ret in episode_returns]
                
                # Update policy and get statistics
                stats = self._update_policy_grpo(relative_rewards)
                
                # Clear group data
                self.current_group_episodes = []
                
                # Clear episode data
                self.current_episode_log_probs = []
                self.current_episode_rewards = []
                
                return stats
        
        # Clear episode data
        self.current_episode_log_probs = []
        self.current_episode_rewards = []
        
        return {
            'policy_loss': 0.0,
            'mean_reward': 0.0,
            'std_reward': 0.0
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
    log_dir = f"/Users/cheencheen/Desktop/rl/RLlama/examples/logs/blackjack_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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