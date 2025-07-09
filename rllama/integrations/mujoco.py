import numpy as np
import gymnasium as gym
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

class MuJoCoEnvWrapper(gym.Wrapper):
    """
    A wrapper for MuJoCo environments that adds additional features.
    
    Features:
    - Domain randomization for robustness
    - Custom observation processing
    - Reward shaping options
    - Episode recording
    """
    
    def __init__(
        self, 
        env_id: str,
        domain_randomization: bool = False,
        observation_noise: float = 0.0,
        reward_shaping: Optional[Dict] = None,
        record_episodes: bool = False,
        record_dir: str = "recordings"
    ):
        """
        Initialize the MuJoCo environment wrapper.
        
        Args:
            env_id: ID of the MuJoCo environment to wrap
            domain_randomization: Whether to apply domain randomization
            observation_noise: Standard deviation of noise to add to observations
            reward_shaping: Dictionary of reward shaping parameters
            record_episodes: Whether to record episodes
            record_dir: Directory to save recordings
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "MuJoCo is not available. Please install it with: "
                "pip install mujoco"
            )
            
        # Create the base environment
        env = gym.make(env_id)
        super().__init__(env)
        
        self.domain_randomization = domain_randomization
        self.observation_noise = observation_noise
        self.reward_shaping = reward_shaping or {}
        self.record_episodes = record_episodes
        self.record_dir = record_dir
        
        # Create recording directory if needed
        if self.record_episodes:
            import os
            os.makedirs(self.record_dir, exist_ok=True)
            
        # Store original parameters for randomization
        if self.domain_randomization:
            self.original_params = self._get_model_parameters()
            
        # Episode data
        self.episode_rewards = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_count = 0
        
    def _get_model_parameters(self) -> Dict:
        """
        Get the current MuJoCo model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        # This is a simplified placeholder - actual implementation would
        # extract physics parameters from the MuJoCo model
        model = self.env.unwrapped.model
        
        params = {
            "body_masses": np.array([model.body_mass[i] for i in range(model.nbody)]),
            "body_inertias": np.array([model.body_inertia[i] for i in range(model.nbody)]),
            "geom_frictions": np.array([model.geom_friction[i] for i in range(model.ngeom)]),
            "dof_damping": np.array([model.dof_damping[i] for i in range(model.nv)]),
            "dof_armature": np.array([model.dof_armature[i] for i in range(model.nv)]),
        }
        
        return params
        
    def _randomize_parameters(self):
        """Apply domain randomization to model parameters."""
        if not self.domain_randomization:
            return
            
        # Access the MuJoCo model
        model = self.env.unwrapped.model
        
        # Randomize body masses (±20%)
        for i in range(model.nbody):
            orig_mass = self.original_params["body_masses"][i]
            model.body_mass[i] = orig_mass * np.random.uniform(0.8, 1.2)
            
        # Randomize frictions (±20%)
        for i in range(model.ngeom):
            orig_friction = self.original_params["geom_frictions"][i]
            model.geom_friction[i] = orig_friction * np.random.uniform(0.8, 1.2)
            
        # Randomize damping (±20%)
        for i in range(model.nv):
            orig_damping = self.original_params["dof_damping"][i]
            model.dof_damping[i] = orig_damping * np.random.uniform(0.8, 1.2)
        
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset the environment with optional domain randomization.
        
        Returns:
            Observation and info dictionary
        """
        # Apply domain randomization
        if self.domain_randomization:
            self._randomize_parameters()
            
        # Reset the base environment
        obs, info = self.env.reset(**kwargs)
        
        # Add observation noise
        if self.observation_noise > 0:
            obs = self._add_noise(obs)
            
        # Reset episode data
        if self.record_episodes:
            self.episode_rewards = []
            self.episode_states = []
            self.episode_actions = []
            
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take a step in the environment with optional modifications.
        
        Args:
            action: Action to take
            
        Returns:
            Observation, reward, terminated, truncated, info
        """
        # Record action if enabled
        if self.record_episodes:
            self.episode_actions.append(action.copy())
            
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add observation noise
        if self.observation_noise > 0:
            obs = self._add_noise(obs)
            
        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, info)
        
        # Record data if enabled
        if self.record_episodes:
            self.episode_rewards.append(shaped_reward)
            self.episode_states.append(obs.copy())
            
            # Save episode if it's done
            if terminated or truncated:
                self._save_episode()
                
        return obs, shaped_reward, terminated, truncated, info
        
    def _add_noise(self, obs: Any) -> Any:
        """
        Add Gaussian noise to observations.
        
        Args:
            obs: Original observation
            
        Returns:
            Noisy observation
        """
        if isinstance(obs, np.ndarray):
            return obs + np.random.normal(0, self.observation_noise, obs.shape)
        elif isinstance(obs, dict):
            noisy_obs = {}
            for key, value in obs.items():
                noisy_obs[key] = self._add_noise(value)
            return noisy_obs
        else:
            return obs
            
    def _shape_reward(self, reward: float, info: Dict) -> float:
        """
        Apply reward shaping based on configuration.
        
        Args:
            reward: Original reward
            info: Info dictionary from environment
            
        Returns:
            Shaped reward
        """
        shaped_reward = reward
        
        # Apply reward scaling
        if "scale" in self.reward_shaping:
            shaped_reward *= self.reward_shaping["scale"]
            
        # Apply reward offset
        if "offset" in self.reward_shaping:
            shaped_reward += self.reward_shaping["offset"]
            
        # Apply energy penalty if configured
        if "energy_penalty" in self.reward_shaping and "energy" in info:
            shaped_reward -= self.reward_shaping["energy_penalty"] * info["energy"]
            
        # Apply smoothness penalty if configured
        if "smoothness_penalty" in self.reward_shaping and len(self.episode_actions) > 1:
            prev_action = self.episode_actions[-2]
            curr_action = self.episode_actions[-1]
            action_diff = np.linalg.norm(curr_action - prev_action)
            shaped_reward -= self.reward_shaping["smoothness_penalty"] * action_diff
            
        return shaped_reward
        
    def _save_episode(self):
        """Save the recorded episode data."""
        if not self.record_episodes:
            return
            
        # Create a unique filename
        import time
        timestamp = int(time.time())
        filename = f"{self.record_dir}/episode_{self.episode_count}_{timestamp}.npz"
        
        # Save the data
        np.savez_compressed(
            filename,
            states=np.array(self.episode_states),
            actions=np.array(self.episode_actions),
            rewards=np.array(self.episode_rewards),
        )
        
        # Increment episode counter
        self.episode_count += 1
        
    def close(self):
        """Close the environment and clean up resources."""
        self.env.close()