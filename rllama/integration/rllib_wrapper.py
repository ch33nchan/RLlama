import gymnasium as gym
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from typing import Dict, Any, Tuple, Optional

from rllama.rewards import RewardComposer, RewardShaper, RewardConfigLoader

class RLlibRllamaEnv(gym.Env, TaskSettableEnv): # Inherit from gym.Env for base API
    """
    An RLlib environment wrapper to integrate RLlama reward shaping.
    This example wraps an existing Gym environment.
    """
    def __init__(self, env_context: EnvContext):
        super().__init__()
        # env_context contains 'worker_index', 'vector_index', and custom config passed to RLlib.
        # The actual environment to wrap should be specified in env_context or created here.
        self.env_name = env_context.get("env_name", "CartPole-v1") # Example: get env name from config
        self.env = gym.make(self.env_name, **env_context.get("env_config", {}))

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        rllama_config_path = env_context.get("rllama_config_path", "path/to/default_rllama_config.yaml")
        self.rllama_config_loader = RewardConfigLoader(config_path=rllama_config_path)
        
        config_dict = self.rllama_config_loader.load_config()
        self.composer: RewardComposer = self.rllama_config_loader.create_composer(config_dict)
        self.shaper: RewardShaper = self.rllama_config_loader.create_shaper(config_dict)
        
        self.pass_full_info_to_rllama = env_context.get("pass_full_info_to_rllama", True)
        self._last_obs = None
        self._last_action = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        if seed is not None: # Pass seed to underlying env
            self._last_obs, info = self.env.reset(seed=seed, options=options)
        else:
            self._last_obs, info = self.env.reset(options=options)

        self._last_action = None
        self.composer.reset()
        self.shaper.reset()
        return self._last_obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        next_obs, base_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        rllama_context = {
            "state": self._last_obs,
            "action": action,
            "next_state": next_obs,
            "base_reward": base_reward,
            "done": done,
            "info": info if self.pass_full_info_to_rllama else {},
            "previous_action": self._last_action,
        }
        
        _, raw_comp_rewards, norm_comp_rewards = self.composer.calculate_reward(**rllama_context)
        
        shaped_reward = self.shaper.shape_reward(
            component_rewards=norm_comp_rewards,
            base_reward=base_reward
        )

        info["rllama_raw_rewards"] = raw_comp_rewards
        info["rllama_normalized_rewards"] = norm_comp_rewards
        info["rllama_weighted_rewards"] = self.shaper.get_last_weighted_rewards()
        info["rllama_total_shaped_reward"] = shaped_reward
        info["rllama_base_reward"] = base_reward
        
        self._last_obs = next_obs
        self._last_action = action

        return next_obs, float(shaped_reward), terminated, truncated, info

    def get_task(self): # For TaskSettableEnv
        if hasattr(self.env, "get_task"):
            return self.env.get_task()
        return None

    def set_task(self, task: Any): # For TaskSettableEnv
        if hasattr(self.env, "set_task"):
            self.env.set_task(task)

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.utils.typing import PolicyID
import yaml

class RLlamaRLlibCallback:
    """
    Ray RLlib callback for RLlama integration
    """
    
    def __init__(self, config_path: str):
        # Load RLlama configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize RLlama components
        from ..core.composer import RewardComposer
        from ..core.shaper import RewardShaper
        
        self.composer = RewardComposer(config.get('composer', {}))
        self.shaper = RewardShaper(config.get('shaper', {}))
        
        # Track metrics
        self.episode_rllama_rewards = []
    
    def on_episode_step(self,
                       *,
                       worker: RolloutWorker,
                       base_env: BaseEnv,
                       episode: MultiAgentEpisode,
                       **kwargs) -> None:
        """Called after each step in an episode"""
        
        # Get the latest step info
        info = episode.last_info_for()
        
        if info and 'prompt' in info and 'response' in info:
            # Calculate RLlama reward
            rllama_reward = self.composer.compose(
                info['prompt'], 
                info['response']
            )
            
            # Store in episode custom metrics
            episode.custom_metrics['rllama_reward'] = rllama_reward
            self.episode_rllama_rewards.append(rllama_reward)
    
    def on_episode_end(self,
                      *,
                      worker: RolloutWorker,
                      base_env: BaseEnv,
                      policies: Dict[PolicyID, Policy],
                      episode: MultiAgentEpisode,
                      **kwargs) -> None:
        """Called at the end of each episode"""
        
        if self.episode_rllama_rewards:
            mean_rllama_reward = sum(self.episode_rllama_rewards) / len(self.episode_rllama_rewards)
            episode.custom_metrics['mean_rllama_reward'] = mean_rllama_reward
            
            # Reset for next episode
            self.episode_rllama_rewards = []

# Example RLlib Config (conceptual):
# config = {
#     "env": RLlibRllamaEnv,
#     "env_config": {
#         "env_name": "YourBaseEnv-v0", # The actual env RLlibRllamaEnv will wrap
#         "rllama_config_path": "path/to/your/rllama_config.yaml",
#         "pass_full_info_to_rllama": True,
#         # ... other configs for YourBaseEnv-v0 ...
#     },
#     # ... other RLlib algo config ...
# }
# from ray.rllib.algorithms.ppo import PPO
# algo = PPO(config=config)
# algo.train()