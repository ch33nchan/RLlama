from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    
    Attributes:
        layers: Sequential layers of the MLP
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None,
    ):
        """
        Initialize an MLP.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            hidden_dims: Dimensions of hidden layers
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
        """
        super().__init__()
        
        layers = []
        dims = (input_dim,) + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
            
        layers.append(nn.Linear(dims[-1], output_dim))
        
        if output_activation is not None:
            layers.append(output_activation)
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.layers(x)


class DiscreteActor(nn.Module):
    """
    Actor network for discrete action spaces.
    
    Attributes:
        net: MLP for computing action logits
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
    ):
        """
        Initialize a discrete actor.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action (number of discrete actions)
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution and sample an action.
        
        Args:
            state: State tensor
            
        Returns:
            Sampled action and log probability
        """
        logits = self.net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of an action.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Log probability of the action
        """
        logits = self.net(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(action)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the action distribution.
        
        Args:
            state: State tensor
            
        Returns:
            Entropy of the action distribution
        """
        logits = self.net(state)
        dist = Categorical(logits=logits)
        return dist.entropy()
    
    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the most likely action.
        
        Args:
            state: State tensor
            
        Returns:
            Most likely action
        """
        logits = self.net(state)
        return torch.argmax(logits, dim=-1)


class ContinuousActor(nn.Module):
    """
    Actor network for continuous action spaces.
    
    Attributes:
        net: MLP for computing action mean
        log_std: Learnable log standard deviation
        action_bounds: Bounds of the action space
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        action_bounds: Optional[np.ndarray] = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """
        Initialize a continuous actor.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: Dimensions of hidden layers
            action_bounds: Bounds of the action space
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        if action_bounds is not None:
            self.register_buffer(
                "action_low", 
                torch.FloatTensor(action_bounds[0])
            )
            self.register_buffer(
                "action_high", 
                torch.FloatTensor(action_bounds[1])
            )
        else:
            self.action_low = None
            self.action_high = None
            
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action distribution and sample an action.
        
        Args:
            state: State tensor
            
        Returns:
            Sampled action and log probability
        """
        mean = self.net(state)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        action = dist.rsample()  # Use reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Apply tanh squashing
        if self.action_low is not None and self.action_high is not None:
            action_tanh = torch.tanh(action)
            # Scale action to be within bounds
            action_scaled = (
                (action_tanh + 1.0) / 2.0 * (self.action_high - self.action_low) + self.action_low
            )
            
            # Compute log probability with squashing correction
            log_prob -= torch.sum(
                torch.log(
                    (self.action_high - self.action_low) * (1 - action_tanh.pow(2)) / 2.0 + 1e-6
                ),
                dim=-1
            )
            
            return action_scaled, log_prob
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of an action.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Log probability of the action
        """
        mean = self.net(state)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # If action is scaled, invert the scaling
        if self.action_low is not None and self.action_high is not None:
            action_tanh = 2.0 * (
                (action - self.action_low) / (self.action_high - self.action_low)
            ) - 1.0
            action = torch.atanh(torch.clamp(action_tanh, -0.999, 0.999))
            
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Apply correction for tanh squashing
        if self.action_low is not None and self.action_high is not None:
            log_prob -= torch.sum(
                torch.log(
                    (self.action_high - self.action_low) * (1 - torch.tanh(action).pow(2)) / 2.0 + 1e-6
                ),
                dim=-1
            )
            
        return log_prob
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of the action distribution.
        
        Args:
            state: State tensor
            
        Returns:
            Entropy of the action distribution
        """
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return torch.sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), dim=-1)
    
    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the mean action (deterministic).
        
        Args:
            state: State tensor
            
        Returns:
            Mean action
        """
        mean = self.net(state)
        
        if self.action_low is not None and self.action_high is not None:
            mean_tanh = torch.tanh(mean)
            return (
                (mean_tanh + 1.0) / 2.0 * (self.action_high - self.action_low) + self.action_low
            )
            
        return mean


class Critic(nn.Module):
    """
    Critic network for estimating value functions.
    
    Attributes:
        net: MLP for computing value
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
    ):
        """
        Initialize a critic.
        
        Args:
            state_dim: Dimension of state
            hidden_dims: Dimensions of hidden layers
        """
        super().__init__()
        
        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate.
        
        Args:
            state: State tensor
            
        Returns:
            Value estimate
        """
        return self.net(state)


class QCritic(nn.Module):
    """
    Q-function critic network for state-action value estimation.
    
    Attributes:
        net: MLP for computing Q-value
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        discrete: bool = False,
    ):
        """
        Initialize a Q-critic.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: Dimensions of hidden layers
            discrete: Whether the action space is discrete
        """
        super().__init__()
        
        self.discrete = discrete
        
        if discrete:
            # For discrete actions, output Q-value for each action
            self.net = MLP(
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_dims=hidden_dims,
            )
        else:
            # For continuous actions, concatenate state and action
            self.net = MLP(
                input_dim=state_dim + action_dim,
                output_dim=1,
                hidden_dims=hidden_dims,
            )
            
    def forward(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Q-value estimate.
        
        Args:
            state: State tensor
            action: Action tensor (only needed for continuous actions)
            
        Returns:
            Q-value estimate
        """
        if self.discrete:
            return self.net(state)
        else:
            if action is None:
                raise ValueError("Action must be provided for continuous Q-function")
            x = torch.cat([state, action], dim=-1)
            return self.net(x)


class ActorCritic(nn.Module):
    """
    Actor-Critic architecture combining policy and value networks.
    
    Attributes:
        actor: Actor network
        critic: Critic network
        discrete: Whether the action space is discrete
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        discrete: bool = True,
        action_bounds: Optional[np.ndarray] = None,
    ):
        """
        Initialize an actor-critic.
        
        Args:
            state_dim: Dimension of state
            action_dim: Dimension of action
            hidden_dims: Dimensions of hidden layers
            discrete: Whether the action space is discrete
            action_bounds: Bounds of the action space (for continuous actions)
        """
        super().__init__()
        
        self.discrete = discrete
        
        if discrete:
            self.actor = DiscreteActor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
            )
        else:
            self.actor = ContinuousActor(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=hidden_dims,
                action_bounds=action_bounds,
            )
            
        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
        )
        
    def forward(
        self, 
        state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor and critic.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action, log_prob = self.actor(state)
        value = self.critic(state)
        
        return action, log_prob, value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate.
        
        Args:
            state: State tensor
            
        Returns:
            Value estimate
        """
        return self.critic(state)
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions and compute log probabilities, entropy, and values.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_probs, entropy, values)
        """
        log_prob = self.actor.get_log_prob(state, action)
        entropy = self.actor.get_entropy(state)
        value = self.critic(state)
        
        return log_prob, entropy, value
    
    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get deterministic action.
        
        Args:
            state: State tensor
            
        Returns:
            Deterministic action
        """
        return self.actor.get_deterministic_action(state)