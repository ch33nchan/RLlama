from typing import Any, Dict, List, Optional, Tuple, Union
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Normal

from rllama.agents.base import BaseAgent
from rllama.core.buffer import ReplayBuffer
from rllama.core.network import Network


class SACActor(Network):
    """
    Actor network for SAC that maps states to action distributions.
    
    Unlike DDPG/TD3, which use deterministic policies, SAC uses a stochastic
    policy that outputs a Gaussian distribution over actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        action_bounds: Optional[np.ndarray] = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """
        Initialize an Actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
            action_bounds: Bounds for the action space (shape: [2, action_dim])
                           where bounds[0] is lower bound and bounds[1] is upper bound
            log_std_min: Minimum value for log standard deviation
            log_std_max: Maximum value for log standard deviation
        """
        super().__init__()
        
        # Save action bounds
        if action_bounds is not None:
            self.action_low = torch.FloatTensor(action_bounds[0])
            self.action_high = torch.FloatTensor(action_bounds[1])
        else:
            self.action_low = -torch.ones(action_dim)
            self.action_high = torch.ones(action_dim)
            
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build the network
        layers = []
        dims = [state_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        
        self.trunk = nn.Sequential(*layers)
        
        # Mean and log_std layers
        self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_linear = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            # Using Xavier initialization
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: State tensor
            
        Returns:
            Tuple of (mean, log_std) of the action distribution
        """
        features = self.trunk(x)
        
        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
        
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy given a state.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob, tanh_mean)
            - action: Sampled action
            - log_prob: Log probability of the action
            - tanh_mean: Mean of the tanh-transformed distribution
        """
        mean, log_std = self(state)
        std = log_std.exp()
        
        # Sample from the Gaussian distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        
        # Scale to action space
        action = self._scale_actions(y_t)
        
        # Calculate log probability of the action
        # (log_prob is the log probability of the original Gaussian distribution,
        # adjusted for the tanh transformation and scaling)
        log_prob = normal.log_prob(x_t)
        
        # Apply tanh squashing correction
        # log_prob = log_prob - torch.log(1 - y_t.pow(2) + 1e-6)
        # For numerical stability, we use log(1-tanh(x)^2) = 2*log(2) - 2*log(exp(x) + exp(-x))
        log_prob = log_prob - 2 * (math.log(2) - x_t - F.softplus(-2 * x_t))
        
        # Sum across action dimensions
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Compute deterministic action (mean)
        tanh_mean = torch.tanh(mean)
        
        return action, log_prob, tanh_mean
        
    def _scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Scale actions from [-1, 1] to the actual action range.
        
        Args:
            actions: Actions in [-1, 1] range from tanh
            
        Returns:
            Scaled actions
        """
        # Make sure action bounds are on the same device as actions
        action_low = self.action_low.to(actions.device)
        action_high = self.action_high.to(actions.device)
        
        # Scale from [-1, 1] to [low, high]
        scaled_actions = 0.5 * (actions + 1.0) * (action_high - action_low) + action_low
        
        return scaled_actions


class SACCritic(Network):
    """
    Critic network for SAC that maps state-action pairs to Q-values.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        """
        Initialize a Critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        # Build Q1 network
        q1_layers = []
        dims = [state_dim + action_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            q1_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                q1_layers.append(nn.ReLU())
        
        self.q1 = nn.Sequential(*q1_layers)
        
        # Build Q2 network
        q2_layers = []
        
        for i in range(len(dims) - 1):
            q2_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                q2_layers.append(nn.ReLU())
        
        self.q2 = nn.Sequential(*q2_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            # Using Xavier initialization
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (Q1, Q2) values
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Get Q1 and Q2 values
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2


class SAC(BaseAgent):
    """
    Soft Actor-Critic agent.
    
    SAC is an off-policy actor-critic algorithm that:
    1. Learns a stochastic policy
    2. Maximizes both reward and entropy
    3. Uses twin critics to mitigate overestimation
    
    Attributes:
        env: Environment the agent interacts with
        actor: Actor network that maps states to action distributions
        critic: Critic network that maps state-action pairs to Q-values
        target_critic: Target critic network for stability
        actor_optimizer: Optimizer for the actor network
        critic_optimizer: Optimizer for the critic network
        alpha: Temperature parameter controlling exploration
        target_entropy: Target entropy for automatic temperature tuning
        log_alpha: Logarithm of alpha for optimization
        alpha_optimizer: Optimizer for alpha
        buffer: Replay buffer for storing experiences
        batch_size: Batch size for updates
        gamma: Discount factor
        tau: Soft update coefficient
    """
    
    def __init__(
        self,
        env: Any,
        actor: Optional[SACActor] = None,
        critic: Optional[SACCritic] = None,
        batch_size: int = 256,
        buffer_size: int = 1000000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        reward_scale: float = 1.0,
        max_grad_norm: Optional[float] = None,
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize a SAC agent.
        
        Args:
            env: Environment the agent interacts with
            actor: Actor network that maps states to action distributions
            critic: Critic network that maps state-action pairs to Q-values
            batch_size: Batch size for updates
            buffer_size: Size of the replay buffer
            actor_lr: Learning rate for the actor optimizer
            critic_lr: Learning rate for the critic optimizer
            alpha_lr: Learning rate for the alpha optimizer
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Initial temperature parameter
            automatic_entropy_tuning: Whether to automatically tune alpha
            reward_scale: Factor to scale rewards by
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run computations on
            verbose: Whether to print verbose output
        """
        super().__init__(env=env, gamma=gamma, device=device, verbose=verbose)
        
        # Check that the environment has a continuous action space
        assert isinstance(self.action_space, gym.spaces.Box), \
            "SAC only works with continuous action spaces"
        
        # Get action space info
        action_dim = self.action_space.shape[0]
        action_bounds = np.vstack([
            self.action_space.low,
            self.action_space.high
        ])
        
        # Get state dimension
        state_dim = np.prod(self.obs_space.shape)
        
        # Create actor if not provided
        if actor is None:
            self.actor = SACActor(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bounds=action_bounds
            ).to(self.device)
        else:
            self.actor = actor.to(self.device)
            
        # Create critic if not provided
        if critic is None:
            self.critic = SACCritic(
                state_dim=state_dim,
                action_dim=action_dim
            ).to(self.device)
        else:
            self.critic = critic.to(self.device)
            
        # Create target critic
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, device=self.device)
        
        # Set hyperparameters
        self.batch_size = batch_size
        self.tau = tau
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        
        # Set up entropy tuning
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        if self.automatic_entropy_tuning:
            # Target entropy is -dim(A) for continuous action spaces
            self.target_entropy = -action_dim
            
            # Optimize log(alpha) instead of alpha for numerical stability
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            # Fixed alpha
            self.alpha = torch.tensor(alpha, device=self.device)
            
        # Initialize episode-specific variables
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        
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
        
    def select_action(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        evaluate: bool = False
    ) -> np.ndarray:
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
            if evaluate:
                # Use mean action for evaluation (deterministic)
                _, _, deterministic_action = self.actor.sample(obs_tensor)
                action = self.actor._scale_actions(deterministic_action)
                return action.cpu().numpy()[0]
            else:
                # Sample from distribution for training (stochastic)
                action, _, _ = self.actor.sample(obs_tensor)
                return action.cpu().numpy()[0]
    
    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """
        Perform soft update of target network parameters.
        
        Args:
            target: Target network
            source: Source network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
        
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
        
        # Take a step in the environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Scale reward for stability
        scaled_reward = reward * self.reward_scale
        
        # Update counters
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += reward  # Track original reward for metrics
        
        # Store the transition with scaled reward
        experience = self.collect_experience(
            obs=self.current_obs,
            action=action,
            next_obs=next_obs,
            reward=scaled_reward,  # Store scaled reward
            done=done,
            info=info
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
        
        # Update networks if buffer has enough samples
        if len(self.buffer) > self.batch_size:
            update_metrics = self.update()
            metrics.update(update_metrics)
            
        return metrics
    
    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update the agent's parameters based on a batch of experiences.
        
        Args:
            batch: Batch of experiences. If None, sample from buffer.
            
        Returns:
            Dictionary of metrics from the update
        """
        # Sample from buffer if batch not provided
        if batch is None:
            batch = self.buffer.sample(self.batch_size)
            
        # Convert batch data to tensors if needed
        states = self._ensure_tensor(batch["obs"])
        actions = self._ensure_tensor(batch["action"])
        rewards = self._ensure_tensor(batch["reward"]).view(-1, 1)  # Ensure column vector
        next_states = self._ensure_tensor(batch["next_obs"])
        dones = self._ensure_tensor(batch["done"], dtype=torch.float).view(-1, 1)  # Ensure column vector
        
        metrics = {}
        
        # ---- Update Critic ----
        with torch.no_grad():
            # Sample next actions and log probs from target actor
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Get target Q-values
            next_q1, next_q2 = self.target_critic(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Include entropy in the target (soft value function)
            next_q = next_q - self.alpha * next_log_probs
            
            # Compute target Q-values
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Get current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Ensure target_q shape matches current_q shape for proper loss calculation
        target_q = target_q.view(current_q1.shape)
        
        # Compute critic losses
        critic_loss1 = F.mse_loss(current_q1, target_q)
        critic_loss2 = F.mse_loss(current_q2, target_q)
        critic_loss = critic_loss1 + critic_loss2
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        metrics["critic_loss"] = critic_loss.item()
        metrics["q1_value"] = current_q1.mean().item()
        metrics["q2_value"] = current_q2.mean().item()
        
        # ---- Update Actor ----
        # Sample actions from the actor
        new_actions, log_probs, _ = self.actor.sample(states)
        
        # Get Q-values for new actions
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        # Compute actor loss (maximize Q - alpha * log_prob)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        metrics["actor_loss"] = actor_loss.item()
        metrics["entropy"] = -log_probs.mean().item()
        
        # ---- Update Alpha (if automatic entropy tuning is enabled) ----
        if self.automatic_entropy_tuning:
            # Compute alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            # Optimize alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update alpha
            self.alpha = self.log_alpha.exp()
            
            metrics["alpha_loss"] = alpha_loss.item()
            metrics["alpha"] = self.alpha.item()
        
        # ---- Update Target Networks ----
        self.soft_update(self.target_critic, self.critic)
            
        return metrics
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state for saving.
        
        Returns:
            Dictionary containing the agent's state
        """
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
        }
        
        if self.automatic_entropy_tuning:
            state["log_alpha"] = self.log_alpha
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
            
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.episodes = state_dict.get("episodes", 0)
        
        if self.automatic_entropy_tuning and "log_alpha" in state_dict:
            self.log_alpha = state_dict["log_alpha"]
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])