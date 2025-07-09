from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rllama.agents.base import BaseAgent
from rllama.core.buffer import ReplayBuffer
from rllama.core.network import Network


class QNetwork(Network):
    """
    Q-Network for DQN.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [64, 64],  # Smaller network to start
    ):
        """
        Initialize a Q-Network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        # Build the network
        layers = []
        dims = [state_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            
        # Output layer
        layers.append(nn.Linear(dims[-1], action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly - critical for stability
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights properly."""
        if isinstance(module, nn.Linear):
            # Smaller initialization for stability
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(x)


class DQN(BaseAgent):
    """
    Deep Q-Network agent.
    
    Attributes:
        env: Environment the agent interacts with
        q_network: Q-Network for estimating Q-values
        target_network: Target Q-Network for stability
        optimizer: Optimizer for updating the Q-Network
        buffer: Replay buffer for storing experiences
        batch_size: Batch size for updates
        gamma: Discount factor
        epsilon: Exploration rate
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Rate of exploration decay
        target_update_freq: Frequency of target network updates
        learning_rate: Learning rate for the optimizer
    """
    
    def __init__(
        self,
        env: Any,
        q_network: Optional[QNetwork] = None,
        batch_size: int = 64,
        buffer_size: int = 10000,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 10,
        double_dqn: bool = False,
        max_grad_norm: Optional[float] = 1.0,  # Default to 1.0 for stability
        reward_scale: float = 1.0,  # Add reward scaling
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize a DQN agent.
        
        Args:
            env: Environment the agent interacts with
            q_network: Q-Network for estimating Q-values
            batch_size: Batch size for updates
            buffer_size: Size of the replay buffer
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of exploration decay
            target_update_freq: Frequency of target network updates
            double_dqn: Whether to use Double DQN
            max_grad_norm: Maximum gradient norm for clipping
            reward_scale: Factor to scale rewards by
            device: Device to run computations on
            verbose: Whether to print verbose output
        """
        super().__init__(env=env, gamma=gamma, device=device, verbose=verbose)
        
        # Create networks if not provided
        if q_network is None:
            state_dim = np.prod(self.obs_space.shape)
            action_dim = self.action_space.n
            self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        else:
            self.q_network = q_network.to(self.device)
            
        # Create target network
        self.target_network = QNetwork(
            self.q_network.network[0].in_features,
            self.q_network.network[-1].out_features,
            [layer.out_features for layer in self.q_network.network if isinstance(layer, nn.Linear)][:-1]
        ).to(self.device)
        self.update_target_network()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize buffer
        self.buffer = ReplayBuffer(capacity=buffer_size, device=self.device)
        
        # Set hyperparameters
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.max_grad_norm = max_grad_norm
        self.reward_scale = reward_scale  # Add reward scaling
        
        # Initialize counters
        self.update_target_counter = 0
        
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
    ) -> int:
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
        
        # Epsilon-greedy action selection
        if not evaluate and np.random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax(dim=1).item()
                
        return action
    
    def update_target_network(self) -> None:
        """Update the target network with the current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
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
            metrics["epsilon"] = self.epsilon
            
            if self.verbose:
                print(f"Episode {self.episodes} finished with reward {self.episode_reward:.2f}")
                
            self.current_obs, _ = self.env.reset()
            self.episode_reward = 0
            self.episode_steps = 0
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update network if buffer has enough samples
        if len(self.buffer) > self.batch_size:
            update_metrics = self.update()
            metrics.update(update_metrics)
            
            # Update target network if needed
            self.update_target_counter += 1
            if self.update_target_counter % self.target_update_freq == 0:
                self.update_target_network()
                self.update_target_counter = 0
            
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
        actions = self._ensure_tensor(batch["action"], dtype=torch.long)
        rewards = self._ensure_tensor(batch["reward"])
        next_states = self._ensure_tensor(batch["next_obs"])
        dones = self._ensure_tensor(batch["done"], dtype=torch.float)
        
        # Make sure tensors have correct shapes
        if len(actions.shape) > 1:
            actions = actions.squeeze()
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values with target network
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Regular DQN: use target network to select and evaluate action
                next_q_values = self.target_network(next_states).max(1)[0]
                
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Use Huber loss for stability instead of smooth_l1_loss
        loss = F.huber_loss(current_q_values, target_q_values, reduction='mean', delta=1.0)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients if specified - critical for stability
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item(),
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state for saving.
        
        Returns:
            Dictionary containing the agent's state
        """
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.episodes = state_dict.get("episodes", 0)
        self.epsilon = state_dict.get("epsilon", self.epsilon_start)