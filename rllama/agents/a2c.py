from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rllama.agents.base import BaseAgent
from rllama.core.network import Network
from rllama.models.policy import ActorCritic


class A2C(BaseAgent):
    """
    Advantage Actor-Critic agent.
    
    Attributes:
        env: Environment the agent interacts with
        policy: Actor-critic policy network
        optimizer: Optimizer for updating the policy
        n_steps: Number of steps to collect before updating
        value_coef: Coefficient for value function loss
        entropy_coef: Coefficient for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
    """
    
    def __init__(
        self,
        env: Any,
        policy: Optional[ActorCritic] = None,
        lr: float = 7e-4,
        n_steps: int = 5,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize an A2C agent.
        
        Args:
            env: Environment the agent interacts with
            policy: Actor-critic policy network
            lr: Learning rate
            n_steps: Number of steps to collect before updating
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            gamma: Discount factor
            device: Device to run computations on
            verbose: Whether to print verbose output
        """
        super().__init__(env=env, gamma=gamma, device=device, verbose=verbose)
        
        # Create policy if not provided
        if policy is None:
            state_dim = np.prod(self.obs_space.shape)
            
            if hasattr(self.action_space, "n"):  # Discrete action space
                action_dim = self.action_space.n
                self.policy = ActorCritic(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    discrete=True
                ).to(self.device)
            else:  # Continuous action space
                action_dim = self.action_space.shape[0]
                action_bounds = np.vstack([
                    self.action_space.low,
                    self.action_space.high
                ])
                self.policy = ActorCritic(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    discrete=False,
                    action_bounds=action_bounds
                ).to(self.device)
        else:
            self.policy = policy.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr, eps=1e-5)
        
        # Set hyperparameters
        self.n_steps = n_steps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize episode-specific variables
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        
        # Initialize rollout buffer
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_values = []
        self.rollout_log_probs = []
        
    def select_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        evaluate: bool = False
    ) -> Union[int, float, np.ndarray]:
        """
        Select an action based on the current observation.
        
        Args:
            obs: Current observation
            evaluate: Whether to select deterministically (for evaluation)
            
        Returns:
            Selected action
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy(obs_tensor)
            
        # Convert to numpy and remove batch dimension
        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().numpy()[0] if log_prob is not None else None
        value = value.cpu().numpy()[0] if value is not None else None
        
        if evaluate:
            # Use mean action for evaluation in continuous action spaces
            if not hasattr(self.action_space, "n"):
                action = self.policy.get_deterministic_action(obs_tensor).cpu().numpy()[0]
        
        return action, log_prob, value
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Returns:
            Dictionary of metrics from the training step
        """
        # Reset environment if needed
        if self.current_obs is None:
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0
        
        # Initialize lists for storing transitions
        self.rollout_obs = []
        self.rollout_actions = []
        self.rollout_rewards = []
        self.rollout_dones = []
        self.rollout_values = []
        self.rollout_log_probs = []
        
        # Collect n_steps transitions
        for _ in range(self.n_steps):
            # Select action
            action, log_prob, value = self.select_action(self.current_obs)
            
            # Store pre-transition data
            self.rollout_obs.append(self.current_obs)
            self.rollout_values.append(value)
            self.rollout_actions.append(action)
            self.rollout_log_probs.append(log_prob)
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update counters
            self.total_steps += 1
            self.episode_steps += 1
            self.episode_reward += reward
            
            # Store post-transition data
            self.rollout_rewards.append(reward)
            self.rollout_dones.append(done)
            
            # Update current observation
            self.current_obs = next_obs
            
            # If episode is done, reset
            if done:
                self.episodes += 1
                if self.verbose:
                    print(f"Episode {self.episodes} finished with reward {self.episode_reward:.2f}")
                    
                self.current_obs, _ = self.env.reset()
                self.episode_reward = 0
                self.episode_steps = 0
                
        # Get value estimate for the final state
        _, _, next_value = self.select_action(self.current_obs)
        
        # Convert lists to tensors
        obs = torch.FloatTensor(np.array(self.rollout_obs)).to(self.device)
        actions = torch.FloatTensor(np.array(self.rollout_actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rollout_rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.rollout_dones)).to(self.device)
        values = torch.FloatTensor(np.array(self.rollout_values)).to(self.device)
        log_probs = torch.FloatTensor(np.array(self.rollout_log_probs)).to(self.device)
        next_value = torch.FloatTensor([next_value]).to(self.device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        
        # Compute GAE
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_val = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * next_non_terminal * gae
            
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
            
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get current predictions
        new_log_probs, entropy, new_values = self.policy.evaluate_actions(obs, actions)
        
        # Compute losses
        policy_loss = -(advantages * new_log_probs).mean()
        value_loss = F.mse_loss(new_values.squeeze(-1), returns)
        entropy_loss = -entropy.mean()
        
        # Compute total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Return metrics
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "episode_reward": self.episode_reward,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state for saving.
        
        Returns:
            Dictionary containing the agent's state
        """
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.policy.load_state_dict(state_dict["policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.episodes = state_dict.get("episodes", 0)