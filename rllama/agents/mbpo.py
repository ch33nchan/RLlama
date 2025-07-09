from typing import Any, Dict, List, Optional, Tuple, Union
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Normal

from rllama.agents.sac import SAC, SACActor, SACCritic
from rllama.agents.base import BaseAgent
from rllama.core.buffer import ReplayBuffer
from rllama.core.network import Network
from rllama.core.experience import Experience


class EnsembleDynamicsModel(Network):
    """
    Ensemble of dynamics models that predict next states and rewards.
    
    Uses an ensemble of neural networks to predict the mean and variance of
    the next state and reward given the current state and action.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dims: List[int] = [200, 200, 200, 200],
        learning_rate: float = 1e-3,
        use_decay: bool = True,
    ):
        """
        Initialize an ensemble of dynamics models.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            ensemble_size: Number of models in the ensemble
            hidden_dims: Dimensions of hidden layers
            learning_rate: Learning rate for the optimizer
            use_decay: Whether to use weight decay in the optimizer
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        
        # Create an ensemble of models
        self.models = nn.ModuleList([
            ProbabilisticDynamicsModel(state_dim, action_dim, hidden_dims)
            for _ in range(ensemble_size)
        ])
        
        # Initialize optimizer with weight decay if requested
        if use_decay:
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=0.000075
            )
        else:
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=learning_rate
            )
            
        # For tracking statistics
        self.mean_state = torch.zeros(state_dim)
        self.std_state = torch.ones(state_dim)
        self.mean_delta = torch.zeros(state_dim)
        self.std_delta = torch.ones(state_dim)
        self.mean_reward = torch.zeros(1)
        self.std_reward = torch.ones(1)
        
    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        deterministic: bool = False,
        return_dist: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next states and rewards using the ensemble.
        
        Args:
            states: Current states
            actions: Actions taken
            deterministic: Whether to make deterministic predictions
            return_dist: Whether to return distribution parameters
            
        Returns:
            Tuple of (next_states, rewards)
        """
        batch_size = states.shape[0]
        
        # Normalize inputs
        states_norm = (states - self.mean_state) / (self.std_state + 1e-8)
        
        # Select a random model for each sample in the batch
        model_indices = torch.randint(
            0, self.ensemble_size, (batch_size,), device=states.device
        )
        
        # For storing results
        next_states = []
        rewards = []
        
        # If returning distribution parameters
        if return_dist:
            means = []
            stds = []
        
        # Process each sample using its assigned model
        for i in range(batch_size):
            model_idx = model_indices[i].item()
            state = states_norm[i:i+1]
            action = actions[i:i+1]
            
            # Get predictions from the model
            delta_mean, delta_std, reward_mean, reward_std = self.models[model_idx](state, action)
            
            # Sample or use mean based on deterministic flag
            if deterministic:
                delta = delta_mean
                reward = reward_mean
            else:
                # Sample from the predicted distribution
                delta = delta_mean + delta_std * torch.randn_like(delta_mean)
                reward = reward_mean + reward_std * torch.randn_like(reward_mean)
            
            # Denormalize delta and convert to next state
            delta = delta * self.std_delta + self.mean_delta
            next_state = states[i:i+1] + delta
            
            # Denormalize reward
            reward = reward * self.std_reward + self.mean_reward
            
            next_states.append(next_state)
            rewards.append(reward)
            
            if return_dist:
                means.append((delta_mean, reward_mean))
                stds.append((delta_std, reward_std))
        
        # Concatenate results
        next_states = torch.cat(next_states, dim=0)
        rewards = torch.cat(rewards, dim=0)
        
        if return_dist:
            return next_states, rewards, means, stds
        else:
            return next_states, rewards
        
    def update(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        """
        Update the ensemble of dynamics models.
        
        Args:
            states: Current states
            actions: Actions taken
            next_states: Next states
            rewards: Rewards received
            
        Returns:
            Dictionary of metrics from the update
        """
        # Calculate state deltas
        deltas = next_states - states
        
        # Update normalization statistics
        self.mean_state = states.mean(dim=0)
        self.std_state = states.std(dim=0) + 1e-8
        self.mean_delta = deltas.mean(dim=0)
        self.std_delta = deltas.std(dim=0) + 1e-8
        self.mean_reward = rewards.mean(dim=0)
        self.std_reward = rewards.std(dim=0) + 1e-8
        
        # Normalize
        states_norm = (states - self.mean_state) / (self.std_state + 1e-8)
        deltas_norm = (deltas - self.mean_delta) / (self.std_delta + 1e-8)
        rewards_norm = (rewards - self.mean_reward) / (self.std_reward + 1e-8)
        
        batch_size = states.shape[0]
        
        # Split data for each model in the ensemble
        # (bootstrapping with replacement)
        model_losses = []
        
        for model in self.models:
            # Sample a batch with replacement
            idxs = np.random.randint(0, batch_size, size=batch_size)
            
            # Get batch for this model
            states_batch = states_norm[idxs]
            actions_batch = actions[idxs]
            deltas_batch = deltas_norm[idxs]
            rewards_batch = rewards_norm[idxs]
            
            # Forward pass
            delta_mean, delta_std, reward_mean, reward_std = model(states_batch, actions_batch)
            
            # Calculate loss (negative log likelihood)
            delta_loss = -Normal(delta_mean, delta_std).log_prob(deltas_batch).mean()
            reward_loss = -Normal(reward_mean, reward_std).log_prob(rewards_batch).mean()
            loss = delta_loss + reward_loss
            
            model_losses.append(loss)
        
        # Calculate total loss
        total_loss = torch.stack(model_losses).mean()
        
        # Update models
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "model_loss": total_loss.item(),
        }


class ProbabilisticDynamicsModel(Network):
    """
    A single probabilistic dynamics model that predicts state transitions and rewards.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [200, 200, 200, 200],
    ):
        """
        Initialize a probabilistic dynamics model.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build shared feature extractor
        layers = []
        input_dim = state_dim + action_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.SiLU())  # SiLU (Swish) activation
            input_dim = dim
            
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layers for delta state prediction
        self.delta_mean = nn.Linear(hidden_dims[-1], state_dim)
        self.delta_logstd = nn.Linear(hidden_dims[-1], state_dim)
        
        # Output layers for reward prediction
        self.reward_mean = nn.Linear(hidden_dims[-1], 1)
        self.reward_logstd = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            # Using fan-in initialization for better gradients
            fan_in = module.weight.data.size(1)
            scale = 1 / np.sqrt(fan_in)
            module.weight.data.uniform_(-scale, scale)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            states: State tensor
            actions: Action tensor
            
        Returns:
            Tuple of (delta_mean, delta_std, reward_mean, reward_std)
        """
        # Concatenate state and action
        inputs = torch.cat([states, actions], dim=1)
        
        # Extract features
        features = self.feature_extractor(inputs)
        
        # Predict delta state
        delta_mean = self.delta_mean(features)
        delta_logstd = self.delta_logstd(features)
        delta_logstd = torch.clamp(delta_logstd, -10, 2)
        delta_std = torch.exp(delta_logstd)
        
        # Predict reward
        reward_mean = self.reward_mean(features)
        reward_logstd = self.reward_logstd(features)
        reward_logstd = torch.clamp(reward_logstd, -10, 2)
        reward_std = torch.exp(reward_logstd)
        
        return delta_mean, delta_std, reward_mean, reward_std


class MBPO(BaseAgent):
    """
    Model-Based Policy Optimization agent.
    
    MBPO alternates between:
    1. Learning a dynamics model from real experience
    2. Using the model to generate synthetic experience
    3. Optimizing a policy with both real and synthetic data
    
    Attributes:
        env: Environment the agent interacts with
        policy: SAC policy for action selection and optimization
        dynamics_model: Ensemble of dynamics models for predicting transitions
        real_buffer: Replay buffer for storing real experiences
        model_buffer: Replay buffer for storing model-generated experiences
        horizon: Planning horizon for model rollouts
        real_ratio: Ratio of real vs model data in policy updates
        updates_per_step: Number of policy updates per environment step
        model_updates_per_step: Number of model updates per environment step
        model_rollout_batch_size: Batch size for model rollouts
        model_retain_epochs: Number of epochs to retain model rollouts
        model_train_frequency: Frequency of model updates
        num_model_rollouts: Number of model rollouts to perform
    """
    
    def __init__(
        self,
        env: Any,
        policy: Optional[SAC] = None,
        dynamics_model: Optional[EnsembleDynamicsModel] = None,
        real_batch_size: int = 256,
        model_batch_size: int = 256,
        real_buffer_size: int = 1000000,
        model_buffer_size: int = 1000000,
        model_ensemble_size: int = 5,
        model_hidden_dims: List[int] = [200, 200, 200, 200],
        model_learning_rate: float = 1e-3,
        horizon: int = 1,
        real_ratio: float = 0.05,
        updates_per_step: int = 20,
        model_updates_per_step: int = 40,
        model_rollout_batch_size: int = 50000,
        model_retain_epochs: int = 1,
        model_train_frequency: int = 250,
        num_model_rollouts: int = 400,
        gamma: float = 0.99,
        device: str = "auto",
        verbose: bool = False,
    ):
        """
        Initialize an MBPO agent.
        
        Args:
            env: Environment the agent interacts with
            policy: SAC policy for action selection and optimization
            dynamics_model: Ensemble of dynamics models for predicting transitions
            real_batch_size: Batch size for real experience
            model_batch_size: Batch size for model experience
            real_buffer_size: Size of the real experience replay buffer
            model_buffer_size: Size of the model experience replay buffer
            model_ensemble_size: Number of models in the dynamics ensemble
            model_hidden_dims: Hidden dimensions for the dynamics model
            model_learning_rate: Learning rate for the dynamics model
            horizon: Planning horizon for model rollouts
            real_ratio: Ratio of real vs model data in policy updates
            updates_per_step: Number of policy updates per environment step
            model_updates_per_step: Number of model updates per environment step
            model_rollout_batch_size: Batch size for model rollouts
            model_retain_epochs: Number of epochs to retain model rollouts
            model_train_frequency: Frequency of model updates
            num_model_rollouts: Number of model rollouts to perform
            gamma: Discount factor
            device: Device to run computations on
            verbose: Whether to print verbose output
        """
        super().__init__(env=env, gamma=gamma, device=device, verbose=verbose)
        
        # Check that the environment has a continuous action space
        assert isinstance(self.action_space, gym.spaces.Box), \
            "MBPO only works with continuous action spaces"
        
        # Get dimensions
        state_dim = np.prod(self.obs_space.shape)
        action_dim = self.action_space.shape[0]
        
        # Create policy if not provided
        if policy is None:
            self.policy = SAC(
                env=env,
                batch_size=real_batch_size,
                buffer_size=real_buffer_size,
                gamma=gamma,
                device=device,
                verbose=False
            )
        else:
            self.policy = policy
            
        # Create dynamics model if not provided
        if dynamics_model is None:
            self.dynamics_model = EnsembleDynamicsModel(
                state_dim=state_dim,
                action_dim=action_dim,
                ensemble_size=model_ensemble_size,
                hidden_dims=model_hidden_dims,
                learning_rate=model_learning_rate
            ).to(self.device)
        else:
            self.dynamics_model = dynamics_model.to(self.device)
            
        # Create model buffer
        self.model_buffer = ReplayBuffer(
            capacity=model_buffer_size,
            device=self.device
        )
        
        # Set hyperparameters
        self.real_batch_size = real_batch_size
        self.model_batch_size = model_batch_size
        self.horizon = horizon
        self.real_ratio = real_ratio
        self.updates_per_step = updates_per_step
        self.model_updates_per_step = model_updates_per_step
        self.model_rollout_batch_size = model_rollout_batch_size
        self.model_retain_epochs = model_retain_epochs
        self.model_train_frequency = model_train_frequency
        self.num_model_rollouts = num_model_rollouts
        
        # To track the age of model-generated samples
        self.model_sample_age = []
        
        # Initialize environment variables
        self.current_obs = None
        self.episode_reward = 0
        self.episode_steps = 0
        
        # For tracking number of model rollouts performed
        self.rollouts_performed = 0
        
    @property
    def real_buffer(self):
        """Access the real experience buffer from the policy."""
        return self.policy.buffer
        
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
        return self.policy.select_action(obs, evaluate)
    
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
        
        # Select action using the policy
        action = self.select_action(self.current_obs)
        
        # Take a step in the environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Update counters
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += reward
        
        # Store the transition in the real buffer
        experience = self.collect_experience(
            obs=self.current_obs,
            action=action,
            next_obs=next_obs,
            reward=reward,
            done=done,
            info=info
        )
        self.real_buffer.add(experience)
        
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
            
        # Train dynamics model periodically
        if self.total_steps % self.model_train_frequency == 0 and len(self.real_buffer) > self.real_batch_size:
            model_metrics = self.train_dynamics_model()
            metrics.update(model_metrics)
            
            # Generate synthetic experience
            rollout_metrics = self.rollout_dynamics_model()
            metrics.update(rollout_metrics)
        
        # Update policy
        if len(self.real_buffer) > self.real_batch_size:
            # Multiple policy updates per environment step
            policy_metrics_list = []
            for _ in range(self.updates_per_step):
                policy_metrics = self.update_policy()
                policy_metrics_list.append(policy_metrics)
            
            # Average policy metrics
            for key in policy_metrics_list[0].keys():
                metrics[key] = np.mean([m[key] for m in policy_metrics_list])
            
        return metrics
    
    def train_dynamics_model(self) -> Dict[str, float]:
        """
        Train the dynamics model on real experience.
        
        Returns:
            Dictionary of metrics from the training
        """
        metrics = {}
        model_update_metrics_list = []
        
        # Multiple model updates
        for _ in range(self.model_updates_per_step):
            # Sample a batch of real experience
            batch = self.real_buffer.sample(self.real_batch_size)
            
            # Convert to tensors with proper type handling
            if isinstance(batch["obs"], torch.Tensor):
                states = batch["obs"].to(dtype=torch.float32, device=self.device)
                actions = batch["action"].to(dtype=torch.float32, device=self.device)
                next_states = batch["next_obs"].to(dtype=torch.float32, device=self.device)
                rewards = batch["reward"].to(dtype=torch.float32, device=self.device).view(-1, 1)
            else:
                states = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
                actions = torch.tensor(batch["action"], dtype=torch.float32, device=self.device)
                next_states = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
                rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device).view(-1, 1)
            
            # Update the model
            model_update_metrics = self.dynamics_model.update(states, actions, next_states, rewards)
            model_update_metrics_list.append(model_update_metrics)
        
        # Average model metrics
        for key in model_update_metrics_list[0].keys():
            metrics[key] = np.mean([m[key] for m in model_update_metrics_list])
            
        return metrics
    
    def rollout_dynamics_model(self) -> Dict[str, float]:
        """
        Generate synthetic experience using the dynamics model.
        
        Returns:
            Dictionary of metrics from the rollout
        """
        metrics = {}
        
        # Clear old model rollouts if we've reached the maximum age
        if len(self.model_sample_age) >= self.model_retain_epochs:
            # Simply clear the buffer and reset ages if there are old samples
            oldest_indices = np.where(np.array(self.model_sample_age) >= self.model_retain_epochs)[0]
            if len(oldest_indices) > 0:
                # Just create a new buffer instead of trying to remove specific indices
                self.model_buffer = ReplayBuffer(
                    capacity=self.model_buffer.capacity,
                    device=self.device
                )
                # Reset ages
                self.model_sample_age = []
        
        # Increment age of existing samples
        self.model_sample_age = [age + 1 for age in self.model_sample_age]
        
        # Start from real states
        if len(self.real_buffer) > 0:
            # Initial states from real experience
            batch = self.real_buffer.sample(self.model_rollout_batch_size)
            if isinstance(batch["obs"], torch.Tensor):
                initial_states = batch["obs"].to(dtype=torch.float32, device=self.device)
            else:
                initial_states = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
            
            # Storage for rollout metrics
            rollout_states = []
            rollout_actions = []
            rollout_next_states = []
            rollout_rewards = []
            rollout_dones = []
            
            # Start from initial states
            states = initial_states
            
            # Rollout for horizon steps
            for h in range(self.horizon):
                # Select actions using the policy
                with torch.no_grad():
                    actions = self.policy.actor.sample(states)[0]
                
                # Predict next states and rewards using the dynamics model
                next_states, rewards = self.dynamics_model(states, actions)
                
                # Set dones to False (we don't model termination)
                dones = torch.zeros(len(states), 1, dtype=torch.bool, device=self.device)
                
                # Add to storage
                rollout_states.append(states)
                rollout_actions.append(actions)
                rollout_next_states.append(next_states)
                rollout_rewards.append(rewards)
                rollout_dones.append(dones)
                
                # Update states for next iteration
                states = next_states
                
                # Break if we've generated enough rollouts
                if h == 0 and len(rollout_states) * len(states) >= self.num_model_rollouts:
                    break
            
            # Convert to numpy and add to model buffer
            total_samples = 0
            for h in range(len(rollout_states)):
                for i in range(len(rollout_states[h])):
                    # Create Experience object and add to buffer
                    experience = Experience(
                        obs=rollout_states[h][i].detach().cpu().numpy(),
                        action=rollout_actions[h][i].detach().cpu().numpy(),
                        next_obs=rollout_next_states[h][i].detach().cpu().numpy(),
                        reward=rollout_rewards[h][i].detach().item(),
                        done=rollout_dones[h][i].detach().item(),
                        info={}
                    )
                    self.model_buffer.add(experience)
                    total_samples += 1
                    
                    # Track age of new samples
                    self.model_sample_age.append(0)
            
            # Update rollout metrics
            metrics["model_rollouts"] = total_samples
            self.rollouts_performed += total_samples
            metrics["total_model_rollouts"] = self.rollouts_performed
            
        return metrics
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update the policy using both real and model-generated experience.
        
        Returns:
            Dictionary of metrics from the update
        """
        metrics = {}
        
        # Determine batch sizes for real and model data
        real_batch_size = int(self.real_batch_size * self.real_ratio)
        model_batch_size = self.real_batch_size - real_batch_size
        
        # Sample real experience
        if real_batch_size > 0 and len(self.real_buffer) >= real_batch_size:
            real_batch = self.real_buffer.sample(real_batch_size)
        else:
            real_batch = None
            
        # Sample model-generated experience
        if model_batch_size > 0 and len(self.model_buffer) >= model_batch_size:
            model_batch = self.model_buffer.sample(model_batch_size)
        else:
            model_batch = None
            
        # Combine batches
        if real_batch is not None and model_batch is not None:
            combined_batch = {
                "obs": np.concatenate([real_batch["obs"], model_batch["obs"]]),
                "action": np.concatenate([real_batch["action"], model_batch["action"]]),
                "next_obs": np.concatenate([real_batch["next_obs"], model_batch["next_obs"]]),
                "reward": np.concatenate([real_batch["reward"], model_batch["reward"]]),
                "done": np.concatenate([real_batch["done"], model_batch["done"]]),
            }
            policy_metrics = self.policy.update(combined_batch)
        elif real_batch is not None:
            policy_metrics = self.policy.update(real_batch)
        elif model_batch is not None:
            policy_metrics = self.policy.update(model_batch)
        else:
            return {}
            
        # Update metrics
        metrics.update(policy_metrics)
        
        # Add source information
        metrics["real_data_ratio"] = real_batch_size / self.real_batch_size if real_batch_size > 0 else 0.0
        metrics["model_buffer_size"] = len(self.model_buffer)
        metrics["real_buffer_size"] = len(self.real_buffer)
        
        return metrics
    
    # Add the required update method
    def update(self, batch: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Update the agent's parameters based on a batch of experiences.
        This method is required by the BaseAgent abstract class.
        
        Args:
            batch: Batch of experiences. If None, update policy with current buffers.
            
        Returns:
            Dictionary of metrics from the update
        """
        if batch is None:
            return self.update_policy()
        else:
            # Add experience to real buffer
            self.real_buffer.add(Experience(**batch))
            
            # Update policy
            return self.update_policy()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the agent's state for saving.
        
        Returns:
            Dictionary containing the agent's state
        """
        return {
            "policy": self.policy.state_dict(),
            "dynamics_model": self.dynamics_model.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "rollouts_performed": self.rollouts_performed,
            "model_sample_age": self.model_sample_age
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the agent's state from a dictionary.
        
        Args:
            state_dict: Dictionary containing the agent's state
        """
        self.policy.load_state_dict(state_dict["policy"])
        self.dynamics_model.load_state_dict(state_dict["dynamics_model"])
        self.total_steps = state_dict.get("total_steps", 0)
        self.episodes = state_dict.get("episodes", 0)
        self.rollouts_performed = state_dict.get("rollouts_performed", 0)
        self.model_sample_age = state_dict.get("model_sample_age", [])