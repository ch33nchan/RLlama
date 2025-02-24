from abc import ABC, abstractmethod
from typing import List, Dict
import copy
from collections import deque
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from trl import (
    PPOTrainer,
    PPOConfig,
    create_reference_model,
)


class Agent(ABC):
    def __init__(
        self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None, algorithm='ppo'
    ):
        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }
        if ppo_config_dict is None:
            ppo_config_dict = {"batch_size": 16, "mini_batch_size": 16}

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.algorithm = algorithm
        self.generate_config_dict = generate_config_dict
        
        # Initialize algorithm-specific components
        self.setup_algorithm()
        
        if self.algorithm == 'ppo':
            self.model_ref = create_reference_model(self.model)
            self.ppo_config = PPOConfig(**ppo_config_dict)
            self.ppo_trainer = PPOTrainer(self.ppo_config, self.model, self.model_ref, self.tokenizer)
            self.current_batch = {"queries": [], "responses": [], "rewards": []}

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []


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
            self.batch_size = 32
            
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
            self.policy_net = nn.Linear(768, 2).to(self.device)
            
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
            
        elif self.algorithm == 'sac':
            # SAC specific parameters
            self.gamma = 0.99
            self.alpha = 0.2
            self.target_entropy = -1.0
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.tau = 0.005
            self.batch_size = 32
            
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

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def format_observation(self, observation: gym.core.ObsType) -> str:
        pass

    @abstractmethod
    def extract_action(self, response: str) -> gym.core.ActType:
        pass

    def llm(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            inputs=inputs.input_ids,
            **{
                key.split("/")[-1]: value
                for key, value in self.generate_config_dict.items()
            }
        )
        outputs = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = outputs[0].split("[/INST]")[-1].strip()

        return response

    def get_state_embedding(self, observation):
        obs_str = self.format_observation(observation)
        inputs = self.tokenizer(obs_str, return_tensors="pt")
        inputs = {k: v.to(self.device).long() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.pretrained_model(**inputs)
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
            state_embedding = hidden_states.mean(dim=1)
        
        return state_embedding.to(self.device)

    def compute_returns_and_advantages(self):
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

    def act(self, observation):
        if hasattr(self, 'algorithm') and self.algorithm != 'ppo':
            state_embedding = self.get_state_embedding(observation)
            
            if self.algorithm == 'dqn':
                with torch.no_grad():
                    if random.random() < self.epsilon:
                        action = random.randint(0, 1)
                    else:
                        q_values = self.model.v_head(state_embedding.float())
                        action = torch.argmax(q_values).item()
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                return action
                
            elif self.algorithm == 'a2c':
                with torch.no_grad():
                    policy_logits = self.policy_net(state_embedding.float())
                    value = self.value_net(state_embedding.float())
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
                
            elif self.algorithm in ['reinforce', 'sac', 'grpo']:
                with torch.no_grad():
                    logits = self.model.v_head(state_embedding.float())
                    action_probs = F.softmax(logits, dim=-1)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    if self.algorithm == 'reinforce':
                        self.saved_log_probs.append(log_prob)
                    elif self.algorithm == 'sac':
                        self.saved_states.append(state_embedding)
                        self.saved_actions.append(action)
                        self.saved_log_probs.append(log_prob)
                    elif self.algorithm == 'grpo':
                        self.current_episode_log_probs.append(log_prob)
                    
                    action = action.clamp(0, 1)
                return action.item()
        
        # Default PPO behavior
        message = self.format_observation(observation)
        self.current_episode_messages += [{"role": "user", "content": message}]
        
        response = self.llm(self.current_episode_messages)
        try:
            action = self.extract_action(response)
        except Exception as e:
            return None
            
        self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action

    def assign_reward(self, reward):
        self.current_episode_rewards.append(reward)

    def format_episode_for_ppo(self, messages, rewards):
        queries, responses = [], []
        for i in range(2, len(messages), 2):
            prompt = self.tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=False, add_generation_prompt=False
            )
            conversation_chunks = prompt.split("[/INST] ")
            query = "[/INST] ".join(conversation_chunks[:-1]) + "[/INST] "
            response = conversation_chunks[-1]

            query = self.tokenizer(query, return_tensors="pt").input_ids[0]
            response = self.tokenizer(response, return_tensors="pt").input_ids[0]

            queries.append(query)
            responses.append(response)

        if all(reward == 0 for reward in rewards[:-1]):
            # if sparse rewards, give equal reward to all conversation turns
            per_turn_reward = rewards[-1] / (len(messages) / 2)
            rewards = [torch.tensor(per_turn_reward, dtype=torch.float16)] * len(
                queries
            )
        else:
            rewards = [torch.tensor(reward, dtype=torch.float16) for reward in rewards]

        return queries, responses, rewards

    def terminate_episode(self, train=True):
        if not train:
            return {}

        if self.algorithm == 'ppo':
            queries, responses, rewards = self.format_episode_for_ppo(
                self.current_episode_messages, self.current_episode_rewards
            )
            
            self.current_batch["queries"].extend(queries)
            self.current_batch["responses"].extend(responses)
            self.current_batch["rewards"].extend(rewards)

            if len(self.current_batch["queries"]) >= self.ppo_config.batch_size:
                train_stats = self.train_batch(
                    self.current_batch["queries"],
                    self.current_batch["responses"],
                    self.current_batch["rewards"],
                )
                return train_stats

        elif self.algorithm == 'dqn':
            if len(self.memory) < self.batch_size:
                return None
                
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states).float()
            next_states = torch.stack(next_states).float()
            actions = torch.tensor(actions, device=self.device)
            rewards = torch.tensor(rewards, device=self.device)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float)
            
            current_q_values = self.model.v_head(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            with torch.no_grad():
                next_q_values = self.target_model.v_head(next_states)
                next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.step_counter % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            return {'loss': loss.item()}

        elif self.algorithm == 'a2c':
            returns, advantages = self.compute_returns_and_advantages()
            if not returns:
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            
            states = torch.stack(self.saved_states)
            actions = torch.stack(self.saved_actions)
            
            policy_logits = self.policy_net(states.float())
            values = self.value_net(states.float())
            
            advantages_tensor = torch.tensor(advantages, device=self.device)
            returns_tensor = torch.tensor(returns, device=self.device)
            
            dist = Categorical(F.softmax(policy_logits, dim=-1))
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            policy_loss = -(log_probs * advantages_tensor).mean()
            value_loss = F.mse_loss(values.squeeze(), returns_tensor)
            
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
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
            if not self.saved_log_probs or not self.saved_rewards:
                return {'policy_loss': 0.0}
            
            returns = []
            R = 0
            for r in reversed(self.saved_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, device=self.device)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for log_prob, R in zip(self.saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            policy_loss = torch.stack(policy_loss).sum()
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            self.saved_log_probs = []
            self.saved_rewards = []
            
            return {'policy_loss': policy_loss.item()}

        elif self.algorithm == 'sac':
            if len(self.memory) < self.batch_size:
                return None
                
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states).float()
            next_states = torch.stack(next_states).float()
            actions = torch.tensor(actions, device=self.device)
            rewards = torch.tensor(rewards, device=self.device)
            dones = torch.tensor(dones, device=self.device, dtype=torch.float)
            
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
            
            logits = self.model.v_head(states)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            new_actions = dist.sample()
            log_probs = dist.log_prob(new_actions)
            
            q1_new = self.q1_net(states).gather(1, new_actions.unsqueeze(1)).squeeze()
            q2_new = self.q2_net(states).gather(1, new_actions.unsqueeze(1)).squeeze()
            q_new = torch.min(q1_new, q2_new)
            
            policy_loss = (self.alpha * log_probs - q_new).mean()
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.q_optimizer.zero_grad()
            (q1_loss + q2_loss).backward()
            self.q_optimizer.step()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
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
            if not self.current_episode_log_probs or not self.current_episode_rewards:
                return {
                    'policy_loss': 0.0,
                    'mean_reward': 0.0,
                    'std_reward': 0.0
                }
            
            episode_return = sum(self.current_episode_rewards)
            
            self.current_group_episodes.append({
                'log_probs': self.current_episode_log_probs.copy(),
                'rewards': [episode_return]
            })
            
            if len(self.current_group_episodes) >= self.group_size:
                episode_returns = [sum(ep['rewards']) for ep in self.current_group_episodes]
                mean_return = sum(episode_returns) / len(episode_returns)
                std_return = torch.std(torch.tensor(episode_returns, device=self.device)).item() if len(episode_returns) > 1 else 0
                
                if std_return > 0:
                    relative_rewards = [(ret - mean_return) / std_return for ret in episode_returns]
                else:
                    relative_rewards = [ret - mean_return for ret in episode_returns]
                
                policy_loss = torch.tensor(0.0, device=self.device)
                for episode, reward in zip(self.current_group_episodes, relative_rewards):
                    if len(episode['log_probs']) > 0:
                        episode_loss = torch.stack([log_prob * (-reward) for log_prob in episode['log_probs']]).sum()
                        policy_loss = policy_loss + episode_loss
                
                if torch.abs(policy_loss) > 1e-8:
                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
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
            
            self.current_episode_log_probs = []
            self.current_episode_rewards = []
            
            return stats

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
        self.current_episode_rewards = []
        
        return {}

    def train_batch(self, batch_queries, batch_responses, batch_rewards):
        if len(batch_queries) > self.ppo_config.batch_size:
            queries = batch_queries[: self.ppo_config.batch_size]
            responses = batch_responses[: self.ppo_config.batch_size]
            rewards = batch_rewards[: self.ppo_config.batch_size]

            # keep the remainder for the next batch
            self.current_batch["queries"] = batch_queries[self.ppo_config.batch_size :]
            self.current_batch["responses"] = batch_responses[
                self.ppo_config.batch_size :
            ]
            self.current_batch["rewards"] = batch_rewards[self.ppo_config.batch_size :]
        else:
            queries, responses, rewards = batch_queries, batch_responses, batch_rewards
            self.current_batch = {"queries": [], "responses": [], "rewards": []}

        train_stats = self.ppo_trainer.step(queries, responses, rewards)
        torch.cuda.empty_cache()

        return train_stats
