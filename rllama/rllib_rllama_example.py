# rllib_rllama_example.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig

# Make sure rllama is in your Python path
# For example, if your script is in an 'examples' directory:
import sys
# Assuming 'examples' is a subdirectory of the project root where 'rllama' package resides
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rllama.integration.rllib_wrapper import RLlibRllamaEnv # The wrapper
from rllama.rewards.base import RewardComponentBase # For custom component
from rllama.utils.config_loader import register_component # To register custom components

# --- 1. Define a Simple Custom Gym Environment (same as SB3 example) ---
class SimplePointEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, target_pos=1.0, max_steps=100):
        super().__init__()
        self.target_pos = np.array([target_pos], dtype=np.float32)
        self.current_pos = np.array([0.0], dtype=np.float32)
        self.max_steps = max_steps
        self.current_step_count = 0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3) # 0: left, 1: stay, 2: right
        self.action_map = [-0.1, 0.0, 0.1]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = np.array([0.0], dtype=np.float32)
        self.current_step_count = 0
        info = {"initial_pos": self.current_pos.copy()}
        return self.current_pos.copy(), info

    def step(self, action):
        self.current_pos += self.action_map[action]
        self.current_step_count += 1
        distance_to_target = np.linalg.norm(self.current_pos - self.target_pos)
        base_reward = -float(distance_to_target)
        terminated = bool(distance_to_target < 0.05)
        truncated = bool(self.current_step_count >= self.max_steps)
        info = {
            "distance_to_target": float(distance_to_target),
            "current_pos": self.current_pos.copy(),
            "target_pos": self.target_pos.copy()
        }
        return self.current_pos.copy(), base_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step_count}, Pos: {self.current_pos[0]:.2f}, Target: {self.target_pos[0]:.2f}")

    def close(self):
        pass

# --- 2. Define Custom RLlama Reward Components (same as SB3 example) ---
class DistanceToTargetReward(RewardComponentBase):
    def __init__(self, name: str = "distance_to_target", weight: float = 1.0, target_key="target_pos", current_key="current_pos"):
        super().__init__(name=name, weight=weight)
        self.target_key = target_key
        self.current_key = current_key

    def calculate_reward(self, context: dict) -> float:
        env_info = context.get("info", {})
        target_pos = env_info.get(self.target_key)
        current_pos = env_info.get(self.current_key)
        if target_pos is not None and current_pos is not None:
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            return -float(distance)
        return 0.0

class ReachedTargetBonus(RewardComponentBase):
    def __init__(self, name: str = "reached_target_bonus", weight: float = 1.0, threshold: float = 0.05, bonus: float = 10.0, target_key="target_pos", current_key="current_pos"):
        super().__init__(name=name, weight=weight)
        self.threshold = threshold
        self.bonus = bonus
        self.target_key = target_key
        self.current_key = current_key
        self._target_reached_this_episode = False

    def calculate_reward(self, context: dict) -> float:
        env_info = context.get("info", {})
        done = context.get("done", False)
        target_pos = env_info.get(self.target_key)
        current_pos = env_info.get(self.current_key)
        reward = 0.0
        if target_pos is not None and current_pos is not None:
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            if distance < self.threshold and not self._target_reached_this_episode:
                reward = self.bonus
                self._target_reached_this_episode = True
        if done: self._target_reached_this_episode = False
        return reward
    
    def reset(self): self._target_reached_this_episode = False

# Register custom components
register_component("DistanceToTargetReward", DistanceToTargetReward)
register_component("ReachedTargetBonus", ReachedTargetBonus)
# Register the SimplePointEnv so RLlibRllamaEnv can make it if needed (though we pass class)
# from ray.tune.registry import register_env
# def env_creator(env_config): return SimplePointEnv(**env_config)
# register_env("SimplePointEnv", env_creator)
# For this example, RLlibRllamaEnv will create SimplePointEnv directly if env_name is passed.

# --- 3. Create RLlama Configuration File (`rllama_config_simple_point.yaml`) ---
# (Ensure this file exists in the same directory or provide the correct path)
# Content is the same as the one from the SB3 example.
# For brevity, we assume it's created. If not, you can add the file creation code here.
CONFIG_FILE_NAME = "rllama_config_simple_point.yaml"

def create_rllama_config_file_if_not_exists(config_file_path):
    if os.path.exists(config_file_path):
        print(f"Using existing RLlama config: {config_file_path}")
        return

    rllama_config_content = """
composer_settings:
  normalization_strategy: 'mean_std'
  norm_window: 100
  epsilon: 1.0e-8
shaper_settings:
  composition_strategy: 'additive'
  default_weight: 0.0
components:
  - name: "distance_penalty"
    class: "DistanceToTargetReward"
    params:
      target_key: "target_pos"
      current_key: "current_pos"
    config:
      weight: 1.0
  - name: "target_bonus"
    class: "ReachedTargetBonus"
    params:
      threshold: 0.05
      bonus: 50.0
      target_key: "target_pos"
      current_key: "current_pos"
    config:
      weight: 1.0
"""
    with open(config_file_path, "w") as f:
        f.write(rllama_config_content)
    print(f"Created RLlama config: {config_file_path}")

# --- 4. Main RLlib Training Logic ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rllama_config_file_abs_path = os.path.join(script_dir, CONFIG_FILE_NAME)
    create_rllama_config_file_if_not_exists(rllama_config_file_abs_path)

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=3) # Adjust num_cpus as needed

    print("Ray initialized.")

    # RLlib PPO Configuration
    # The RLlibRllamaEnv will be instantiated by RLlib workers.
    # It needs to know how to create the underlying SimplePointEnv.
    # We pass 'env_name' and 'env_config' for SimplePointEnv to RLlibRllamaEnv's env_context.
    
    # Method 1: If SimplePointEnv is registered with RLlib via ray.tune.registry.register_env
    # env_name_for_rllib = "SimplePointEnv" 
    # Method 2: Or, if RLlibRllamaEnv is modified to accept a class directly (not current design)
    # For current RLlibRllamaEnv, it expects 'env_name' to be a string for gym.make()

    # To make it work with the current RLlibRllamaEnv, we need to ensure SimplePointEnv
    # can be created by gym.make(). This usually means it needs to be installed or
    # registered in a way gym recognizes it.
    # For a local script, a simpler way is to modify RLlibRllamaEnv to accept
    # the env class or an instance directly, or ensure gym.make("local_module:SimplePointEnv") works.

    # For this example, let's assume gym.make can find SimplePointEnv if it's in the path
    # and we use a dummy name that gym.make would use if it were a registered entry point.
    # A more robust way for custom envs not installed as packages is to use a creator function.
    # However, our RLlibRllamaEnv currently uses gym.make(env_context.get("env_name", ...))

    # Let's adjust RLlibRllamaEnv slightly for easier use with local, non-installed custom envs
    # (This would be a change in rllama/integration/rllib_wrapper.py)
    # For now, let's assume we can pass enough info for it to construct SimplePointEnv.
    # The current RLlibRllamaEnv uses gym.make(self.env_name, **env_context.get("env_config", {}))
    # So, if SimplePointEnv is in the python path, gym.make might work if we give it a name
    # that implies its module. This is tricky without proper registration.

    # A common pattern for custom envs in RLlib:
    # from ray.tune.registry import register_env
    # def _env_creator(env_config): return SimplePointEnv(**env_config)
    # register_env("MySimplePointEnv", _env_creator)
    # Then use "MySimplePointEnv" in the config.

    # For this script, to keep it self-contained without modifying RLlibRllamaEnv for now,
    # we'll rely on the fact that SimplePointEnv is defined in the same script's scope.
    # RLlib's default environment creation might pick it up if we pass the class.
    # Let's try configuring RLlib to use the RLlibRllamaEnv class directly.

    config = (
        PPOConfig()
        .environment(
            env=RLlibRllamaEnv, # Pass the wrapper class directly
            env_config={ # This is the env_context for RLlibRllamaEnv
                "env_name": "SimplePointEnv", # A name for the underlying env
                                             # RLlibRllamaEnv will try gym.make(this_name, **underlying_env_config)
                                             # This will only work if SimplePointEnv is discoverable by gym.make
                                             # (e.g. if it's in an installed package or via entry points)
                                             # For a script, this is often an issue.
                
                # To make SimplePointEnv work here without installation:
                # One workaround: modify RLlibRllamaEnv to accept `env_class`
                # "env_class": SimplePointEnv, # Hypothetical modification
                "underlying_env_config": { # Config for SimplePointEnv
                    "target_pos": 1.0,
                    "max_steps": 200
                },
                "rllama_config_path": rllama_config_file_abs_path,
                "pass_full_info_to_rllama": True,
            }
        )
        .framework("torch") # or "tf"
        .rollouts(num_rollout_workers=1) # For local testing
        .training(model={"fcnet_hiddens": [64, 64]})
        .resources(num_gpus=0) # Adjust if you have GPUs
    )
    
    # Hack to make gym.make("SimplePointEnv") work within RLlib workers for this script:
    # This is not ideal for production but helps for a self-contained example.
    # RLlib serializes the env_config. If 'SimplePointEnv' is just a string,
    # workers need to be able to resolve it.
    # A better way is to register the env with RLlib.
    from ray.tune.registry import register_env
    def env_creator(env_config_for_simple_point):
        # env_config_for_simple_point here is what we put in "underlying_env_config"
        return SimplePointEnv(**env_config_for_simple_point)
    register_env("SimplePointEnv", env_creator) # Now gym.make("SimplePointEnv") will work in workers

    print("Building PPO agent...")
    try:
        agent = config.build()
        print("PPO agent built.")

        print("\nTraining PPO agent...")
        for i in range(5): # Train for a few iterations
            result = agent.train()
            print(f"Iteration {i+1}:")
            print(f"  episode_reward_mean: {result['episode_reward_mean']:.2f}")
            print(f"  timesteps_total: {result['timesteps_total']}")
            # You can access custom metrics from info dict if you log them via callbacks
            if 'hist_stats' in result and 'rllama_total_shaped_reward' in result['hist_stats']:
                 print(f"  Mean shaped reward (hist): {np.mean(result['hist_stats']['rllama_total_shaped_reward']):.2f}")


        print("Training finished.")

        # Test the trained agent (optional)
        print("\nTesting trained agent (conceptual):")
        # env_instance = RLlibRllamaEnv(config.env_config) # Create a local instance for testing
        # obs, info = env_instance.reset()
        # for _ in range(50):
        #     action = agent.compute_single_action(obs)
        #     obs, reward, terminated, truncated, info = env_instance.step(action)
        #     print(f"  Obs: {obs[0]:.2f}, Shaped Rew: {reward:.2f}, Base Rew: {info.get('rllama_base_reward',0):.2f}, Done: {terminated or truncated}")
        #     if "rllama_weighted_rewards" in info: print(f"    RLlama Weighted: {info['rllama_weighted_rewards']}")
        #     if terminated or truncated: break
        # env_instance.close()

    except Exception as e:
        print(f"An error occurred during RLlib agent build or training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'agent' in locals() and agent:
            agent.stop()
        ray.shutdown()
        print("Ray shutdown.")
        # if os.path.exists(rllama_config_file_abs_path):
        #     os.remove(rllama_config_file_abs_path) # Clean up config file
        #     print(f"Cleaned up '{rllama_config_file_abs_path}'.")

if __name__ == "__main__":
    main()