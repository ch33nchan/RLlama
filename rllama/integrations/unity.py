from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from rllama.environments.base import BaseEnvironment


def wrap_unity(env: Any) -> gym.Env:
    """
    Wrap a Unity ML-Agents environment to make it compatible with RLlama.
    
    Args:
        env: Unity ML-Agents environment to wrap
        
    Returns:
        Wrapped environment
        
    Raises:
        ImportError: If Unity ML-Agents is not installed
    """
    try:
        import mlagents
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    except ImportError:
        raise ImportError(
            "Unity ML-Agents is not installed. "
            "Please install it with `pip install mlagents`."
        )
        
    return UnityWrapper(env)


class UnityWrapper(gym.Wrapper):
    """
    Wrapper for Unity ML-Agents environments to make them compatible with RLlama.
    
    Attributes:
        env: Unity ML-Agents environment
        behavior_name: Name of the behavior
    """
    
    def __init__(self, env: Any):
        """
        Initialize a Unity ML-Agents wrapper.
        
        Args:
            env: Unity ML-Agents environment to wrap
        """
        super().__init__(env)
        
        # Get behavior name and spec
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]
        
        # Define observation and action spaces
        self._define_spaces()
        
    def _define_spaces(self) -> None:
        """Define observation and action spaces based on the Unity environment."""
        # Get observation spec
        obs_spec = self.spec.observation_specs
        
        # Create observation space
        if len(obs_spec) == 1:
            # Single observation
            obs_shape = obs_spec[0].shape
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_shape,
                dtype=np.float32
            )
        else:
            # Multiple observations
            obs_spaces = {}
            for i, spec in enumerate(obs_spec):
                obs_spaces[f"obs_{i}"] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=spec.shape,
                    dtype=np.float32
                )
            self.observation_space = gym.spaces.Dict(obs_spaces)
            
        # Get action spec
        action_spec = self.spec.action_spec
        
        # Create action space
        if action_spec.continuous_size > 0:
            # Continuous action space
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_spec.continuous_size,),
                dtype=np.float32
            )
        else:
            # Discrete action space
            self.action_space = gym.spaces.MultiDiscrete(
                [branch for branch in action_spec.discrete_branches]
            )
            
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the Unity environment."""
        if seed is not None:
            # Unity doesn't support seeding directly, but we could reset the environment
            # with a specific seed if the environment supports it
            pass
            
        self.env.reset()
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        
        # Get the first agent's observation
        agent_id = decision_steps.agent_id[0]
        obs = self._extract_obs(decision_steps, agent_id)
        
        return obs, {}
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the Unity environment."""
        # Get current decision steps
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        # Convert action to Unity format
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action
            unity_action = action
        else:
            # Discrete action
            unity_action = np.zeros(len(self.action_space.nvec), dtype=np.int32)
            unity_action[:len(action)] = action
            
        # Set actions for all agents
        self.env.set_actions(self.behavior_name, unity_action.reshape(1, -1))
        
        # Step the environment
        self.env.step()
        
        # Get new decision steps and terminal steps
        new_decision_steps, new_terminal_steps = self.env.get_steps(self.behavior_name)
        
        # Get the agent ID
        agent_id = decision_steps.agent_id[0]
        
        # Check if the agent terminated
        if agent_id in new_terminal_steps:
            steps = new_terminal_steps
            terminated = True
            truncated = False
        else:
            steps = new_decision_steps
            terminated = False
            truncated = False
            
        # Extract observation, reward, and info
        obs = self._extract_obs(steps, agent_id)
        reward = steps.reward[steps.agent_id_to_index[agent_id]]
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _extract_obs(self, steps: Any, agent_id: int) -> Any:
        """
        Extract observation for a specific agent.
        
        Args:
            steps: Decision steps or terminal steps
            agent_id: ID of the agent
            
        Returns:
            Observation for the agent
        """
        idx = steps.agent_id_to_index[agent_id]
        
        if len(steps.obs) == 1:
            # Single observation
            return steps.obs[0][idx]
        else:
            # Multiple observations
            return {f"obs_{i}": obs[idx] for i, obs in enumerate(steps.obs)}