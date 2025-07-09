from typing import Any, Dict, List, Optional, Tuple, Union
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from rllama.agents.base import BaseAgent
from rllama.core.buffer import ReplayBuffer
from rllama.core.network import Network


class Actor(Network):
    """
    Actor network for TD3 that maps states to deterministic actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        action_bounds: Optional[np.ndarray] = None,
    ):
        """
        Initialize an Actor network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
            action_bounds: Bounds for the action space (shape: [2, action_dim])
                           where bounds[0] is lower bound and bounds[1] is upper bound
        """
        super().__init__()
        
        # Save action bounds
        if action_bounds is not None:
            self.action_low = torch.FloatTensor(action_bounds[0])
            self.action_high = torch.FloatTensor(action_bounds[1])
        else:
            self.action_low = -torch.ones(action_dim)
            self.action_high = torch.ones(action_dim)
            
        # Build the network
        layers = []
        dims = [state_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            
        # Output layer with tanh activation to bound actions
        layers.append(nn.Linear(dims[-1], action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            # Using Xavier initialization
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: State tensor
            
        Returns:
            Action tensor
        """
        # Get raw actions in [-1, 1] range from tanh
        raw_actions = self.network(x)
        
        # Scale to the actual action range
        scaled_actions = self._scale_actions(raw_actions)
        
        return scaled_actions
        
    def _scale_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        """
        Scale actions from [-1, 1] to the actual action range.
        
        Args:
            raw_actions: Actions in [-1, 1] range
            
        Returns:
            Scaled actions
        """
        # Make sure action bounds are on the same device as actions
        action_low = self.action_low.to(raw_actions.device)
        action_high = self.action_high.to(raw_actions.device)
        
        # Scale from [-1, 1] to [low, high]
        scaled_actions = 0.5 * (raw_actions + 1.0) * (action_high - action_low) + action_low
        
        return scaled_actions


class Critic(Network):
    """
    Critic network for TD3 that maps state-action pairs to Q-values.
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
        
        # First layer processes only the state
        self.state_layer = nn.Linear(state_dim, hidden_dims[0])
        self.state_activation = nn.ReLU()
        
        # Subsequent layers process state and action together
        layers = []
        dims = [hidden_dims[0] + action_dim] + hidden_dims[1:]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            # Using Xavier initialization
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        # Process state
        state_features = self.state_activation(self.state_layer(state))
        
        # Concatenate state features and action
        x = torch.cat([state_features, action], dim=1)
        
        # Process state-action pair
        q_value = self.network(x)
        
        return q_value


class TD3(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient agent.
    
    TD3 improves upon DDPG with three key modifications:
    1. Twin critics to reduce overestimation bias
    2. Delayed policy updates
    3. Target policy smoothing (adding noise to target actions)
    
    Attributes:
        env: Environment the agent interacts with
        actor: Actor network that maps states to actions
        critic1: First critic network that maps state-action pairs to Q-values
        critic2: Second critic network that maps state-action pairs to Q-values
        target_actor: Target actor network for stability
        target_critic1: Target first critic network for stability
        target_critic2: Target second critic network for stability
        actor_optimizer: Optimizer for the actor network
        critic1_optimizer: Optimizer for the first critic network
        critic2_optimizer: Optimizer for the second critic network
        buffer: Replay buffer for storing experiences
        batch_size: Batch size for updates
        gamma: Discount factor
        tau: Soft update coefficient
        policy_noise: Noise added to target actions for smoothing
        noise_clip: Maximum value of policy noise
        policy_delay: Number of critic updates per actor update
    """
    
    def __init__(
        self,
        env: Any,
        actor: Optional[Actor] = None,
        critic1: Optional[Critic] = None,
        critic2: Optional[Critic] = None,
        batch_size: int = 100,
        buffer_size: int = 1000000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        exploration_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        reward_scale: float = 1.0,
        max_grad_norm: Optional[float] = None,
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize a TD3 agent.
        
        Args:
            env: Environment the agent interacts with
            actor: Actor network that maps states to actions
            critic1: First critic network that maps state-action pairs to Q-values
            critic2: Second critic network that maps state-action pairs to Q-values
            batch_size: Batch size for updates
            buffer_size: Size of the replay buffer
            actor_lr: Learning rate for the actor optimizer
            critic_lr: Learning rate for the critic optimizer
            gamma: Discount factor
            tau: Soft update coefficient
            exploration_noise: Scale of exploration noise
            policy_noise: Noise added to target actions for smoothing
            noise_clip: Maximum value of policy noise
            policy_delay: Number of critic updates per actor update
            reward_scale: Factor to scale rewards by
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run computations on
            verbose: Whether to print verbose output
        """
        super().__init__(env=env, gamma=gamma, device=device, verbose=verbose)
        
        # Check that the environment has a continuous action space
        assert isinstance(self.action_space, gym.spaces.Box), \
            "TD3 only works with continuous action spaces"
        
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
            self.actor = Actor(
                state_dim=state_dim,
                action_dim=action_dim,
                action_bounds=action_bounds
            ).to(self.device)
        else:
            self.actor = actor.to(self.device)
            
        # Create critics if not provided
        if critic1 is None:
            self.critic1 = Critic(
                state_dim=state_dim,
                action_dim=action_dim
            ).to(self.device)
        else:
            self.critic1 = critic1.to(self.device)
            
        if critic2 is None:
            self.critic2 = Critic(
                state_dim=state_dim,
                action_dim=action_dim
            ).to(self.device)
        else:
            self.critic2 = critic2.to(self.device)
            
        # Create target networks
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        # Initialize buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, device=self.device)
        
        # Set hyperparameters
        self.batch_size = batch_size
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        
        # Initialize episode-specific variables
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        
        # Initialize counters
        self.update_counter = 0
        
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
        
        # Get action from actor network
        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy()[0]
            
        # Add exploration noise if not evaluating
        if not evaluate:
            # Add Gaussian noise
            action += np.random.normal(0, self.exploration_noise, size=action.shape)
                
            # Clip action to valid range
            action = np.clip(
                action,
                self.action_space.low,
                self.action_space.high
            )
                
        return action
    
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
        rewards = self._ensure_tensor(batch["reward"])
        next_states = self._ensure_tensor(batch["next_obs"])
        dones = self._ensure_tensor(batch["done"], dtype=torch.float)
        
        metrics = {}
        
        # ---- Update Critics ----
        with torch.no_grad():
            # Select action from target actor and add clipped noise for smoothing
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            next_actions = self.target_actor(next_states)
            next_actions = next_actions + noise
            next_actions = torch.clamp(
                next_actions,
                torch.tensor(self.action_space.low, device=self.device),
                torch.tensor(self.action_space.high, device=self.device)
            )
            
            # Get minimum Q-value from both target critics
            next_q1 = self.target_critic1(next_states, next_actions)
            next_q2 = self.target_critic2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Compute target Q-values
            target_q = rewards + (1 - dones) * self.gamma * next_q.squeeze()
            
        # Update first critic
        current_q1 = self.critic1(states, actions).squeeze()
        critic1_loss = F.mse_loss(current_q1, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()
        
        # Update second critic
        current_q2 = self.critic2(states, actions).squeeze()
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic2_optimizer.step()
        
        metrics["critic1_loss"] = critic1_loss.item()
        metrics["critic2_loss"] = critic2_loss.item()
        metrics["q1_value"] = current_q1.mean().item()
        metrics["q2_value"] = current_q2.mean().item()
        
        # ---- Delayed Update of Actor and Target Networks ----
        self.update_counter += 1
        if self.update_counter % self.policy_delay == 0:
            # Update Actor
            actor_actions = self.actor(states)
            actor_loss = -self.critic1(states, actor_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Soft update target networks
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)
            
            metrics["actor_loss"] = actor_loss.item()
            metrics["policy_updated"] = 1.0
        else:
            metrics["policy_updated"] = 0.0
            
        return metrics
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state for saving.
        
        Returns:
            Dictionary containing the agent's state
        """
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "update_counter": self.update_counter,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic1.load_state_dict(state_dict["target_critic1"])
        self.target_critic2.load_state_dict(state_dict["target_critic2"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state_dict["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state_dict["critic2_optimizer"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.episodes = state_dict.get("episodes", 0)
        self.update_counter = state_dict.get("update_counter", 0)