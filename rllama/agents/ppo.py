from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rllama.agents.base import BaseAgent
from rllama.core.buffer import EpisodeBuffer
from rllama.core.network import Network
from rllama.models.policy import ActorCritic


class PPO(BaseAgent):
    """
    Proximal Policy Optimization agent.
    
    Attributes:
        env: Environment the agent interacts with
        policy: Actor-critic policy network
        buffer: Buffer for storing trajectories
        optimizer: Optimizer for updating the policy
        clip_ratio: PPO clip ratio
        value_coef: Coefficient for value function loss
        entropy_coef: Coefficient for entropy bonus
        max_grad_norm: Maximum gradient norm for clipping
        update_epochs: Number of epochs to update policy per batch
        batch_size: Batch size for updates
        gae_lambda: GAE lambda parameter
    """
    
    def __init__(
        self,
        env: Any,
        policy: Optional[ActorCritic] = None,
        lr: float = 3e-4,
        learning_rate: Optional[float] = None,  # Added for compatibility
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize a PPO agent.
        
        Args:
            env: Environment the agent interacts with
            policy: Actor-critic policy network
            lr: Learning rate
            learning_rate: Alternative name for learning rate (if provided, overrides lr)
            clip_ratio: PPO clip ratio
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs to update policy per batch
            batch_size: Batch size for updates
            buffer_size: Size of the replay buffer
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
            device: Device to run computations on
            verbose: Whether to print verbose output
        """
        super().__init__(env=env, gamma=gamma, device=device, verbose=verbose)
        
        # Use learning_rate if provided, otherwise use lr
        if learning_rate is not None:
            lr = learning_rate
        
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
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize buffer
        self.buffer = EpisodeBuffer(capacity=buffer_size, device=self.device)
        
        # Set hyperparameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        
        # Initialize episode-specific variables
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        
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
        
        return action
    
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
        
        # Select action
        action = self.select_action(self.current_obs)
        
        # Get log probability and value estimate for the action
        obs_tensor = torch.FloatTensor(self.current_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, log_prob, value = self.policy(obs_tensor)
            log_prob = log_prob.cpu().numpy()[0]
            value = value.cpu().numpy()[0]
        
        # Take a step in the environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Update counters
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += reward
        
        # Store the transition
        experience = self.collect_experience(
            obs=self.current_obs,
            action=action,
            next_obs=next_obs,
            reward=reward,
            done=done,
            info=info,
            log_prob=log_prob,
            value=value
        )
        self.buffer.add(experience)
        
        # Update current observation
        self.current_obs = next_obs
        
        # If episode is done, reset
        metrics = {}
        if done:
            self.episodes += 1
            metrics["episode_reward"] = self.episode_reward
            metrics["episode_length"] = self.episode_steps
            
            if self.verbose:
                print(f"Episode {self.episodes} finished with reward {self.episode_reward:.2f}")
                
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
        
        # Update policy if buffer is full
        if len(self.buffer) >= self.buffer.capacity:
            update_metrics = self._update_policy()
            metrics.update(update_metrics)
            self.buffer.clear()
            
        return metrics
    
    def _ensure_tensor(self, data: Any, dtype=torch.float32) -> torch.Tensor:
        """
        Ensure data is a tensor.
        
        Args:
            data: Data to convert
            dtype: Data type for the tensor
            
        Returns:
            Tensor
        """
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=dtype, device=self.device)
        elif isinstance(data, list):
            return torch.tensor(data, dtype=dtype, device=self.device)
        else:
            return torch.tensor([data], dtype=dtype, device=self.device)
    
    def _compute_advantages(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE).
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Tuple of advantages and returns tensors
        """
        # Ensure all data is in tensor format
        obs = self._ensure_tensor(batch["obs"])
        next_obs = self._ensure_tensor(batch["next_obs"])
        rewards = self._ensure_tensor(batch["reward"])
        # Convert boolean tensor to float tensor before subtraction
        dones = self._ensure_tensor(batch["done"]).float()  
        values = self._ensure_tensor(batch["value"])
        
        # Make sure values is 1D
        if len(values.shape) > 1:
            values = values.squeeze()
        
        # Compute value of next observations
        with torch.no_grad():
            next_values = self.policy.get_value(next_obs).squeeze(-1)
            
        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute GAE
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _update_policy(self) -> Dict[str, float]:
        """
        Update the policy using PPO.
        
        Returns:
            Dictionary of metrics from the update
        """
        # Sample all experiences from buffer
        batch_raw = self.buffer.get_all()
        
        # Create a dictionary with tensors
        batch = {}
        for key, value in batch_raw.items():
            if key != "info":  # Skip info dictionary
                batch[key] = self._ensure_tensor(value)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(batch)
        
        # Ensure all tensors have the right shape
        if len(advantages.shape) > 1:
            advantages = advantages.squeeze()
        if len(returns.shape) > 1:
            returns = returns.squeeze()
        
        # Store metrics for reporting
        total_metrics = {}
        
        # Update policy for multiple epochs
        for epoch in range(self.update_epochs):
            # Generate random indices
            num_samples = batch["obs"].shape[0]
            indices = torch.randperm(num_samples).to(self.device)
            
            # Update in batches
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]
                
                # Get batch data
                mb_obs = batch["obs"][batch_idx]
                mb_actions = batch["action"][batch_idx]
                mb_old_log_probs = batch["log_prob"][batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns[batch_idx]
                
                # Make sure all tensors have the right shape
                if len(mb_advantages.shape) > 1:
                    mb_advantages = mb_advantages.squeeze()
                if len(mb_returns.shape) > 1:
                    mb_returns = mb_returns.squeeze()
                if len(mb_old_log_probs.shape) > 1:
                    mb_old_log_probs = mb_old_log_probs.squeeze()
                
                # Get current log probs and values
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(mb_obs, mb_actions)
                
                # Ensure log probs and values have the right shape
                if len(new_log_probs.shape) > 1:
                    new_log_probs = new_log_probs.squeeze()
                
                # Make sure new_values is the right shape
                new_values = new_values.squeeze()
                
                # Print shapes for debugging (temporarily)
                # print(f"mb_returns shape: {mb_returns.shape}, new_values shape: {new_values.shape}")
                
                # Calculate ratios and surrogates for PPO update
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = -entropy.mean()
                
                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store metrics for this batch
                total_metrics = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy_loss.item(),
                }
        
        return total_metrics
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the agent's parameters based on a batch of experiences.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of metrics from the update
        """
        # Ensure all data is in tensor format
        processed_batch = {}
        for key, value in batch.items():
            if key != "info":  # Skip info dictionary
                processed_batch[key] = self._ensure_tensor(value)
            
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(processed_batch)
        
        # Ensure all tensors have the right shape
        if len(advantages.shape) > 1:
            advantages = advantages.squeeze()
        if len(returns.shape) > 1:
            returns = returns.squeeze()
        
        # Store metrics for reporting
        total_metrics = {}
        
        # Update policy for multiple epochs
        for epoch in range(self.update_epochs):
            # Generate random indices
            num_samples = processed_batch["obs"].shape[0]
            indices = torch.randperm(num_samples).to(self.device)
            
            # Update in batches
            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]
                
                # Get batch data
                mb_obs = processed_batch["obs"][batch_idx]
                mb_actions = processed_batch["action"][batch_idx]
                mb_old_log_probs = processed_batch["log_prob"][batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns[batch_idx]
                
                # Make sure all tensors have the right shape
                if len(mb_advantages.shape) > 1:
                    mb_advantages = mb_advantages.squeeze()
                if len(mb_returns.shape) > 1:
                    mb_returns = mb_returns.squeeze()
                if len(mb_old_log_probs.shape) > 1:
                    mb_old_log_probs = mb_old_log_probs.squeeze()
                
                # Get current log probs and values
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(mb_obs, mb_actions)
                
                # Ensure log probs and values have the right shape
                if len(new_log_probs.shape) > 1:
                    new_log_probs = new_log_probs.squeeze()
                
                # Make sure new_values is the right shape
                new_values = new_values.squeeze()
                
                # Calculate ratios and surrogates for PPO update
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                
                # Calculate losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = -entropy.mean()
                
                # Calculate total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store metrics for this epoch
                total_metrics = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy_loss.item(),
                }
        
        return total_metrics
    
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