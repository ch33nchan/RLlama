# examples/sb3_blackjack_example.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import sys
import os

# Ensure RLlama is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from rllama.integration.sb3_wrapper import SB3RllamaRewardWrapper

# --- Configuration ---
# We can reuse an existing simple config file.
RLLAMA_CONFIG_PATH = "examples/optimizer_config.yaml"
TOTAL_TIMESTEPS = 25000

def main():
    """Main function to run the SB3 training."""
    print("--- Starting Stable Baselines 3 + RLlama Blackjack Example ---")

    # 1. Create the base Gymnasium environment
    base_env = gym.make("Blackjack-v1")
    print("Base Blackjack environment created.")

    # 2. Wrap the environment with our RLlama Reward Wrapper
    # This single line injects the entire RLlama reward system.
    try:
        wrapped_env = SB3RllamaRewardWrapper(
            env=base_env,
            rllama_config_path=RLLAMA_CONFIG_PATH
        )
        print(f"Environment wrapped with RLlama using config: {RLLAMA_CONFIG_PATH}")
    except Exception as e:
        print(f"Error wrapping environment: {e}")
        return

    # 3. (Optional but recommended) Check if the wrapped environment is SB3 compatible
    print("Checking environment compatibility with Stable Baselines 3...")
    check_env(wrapped_env)
    print("✅ Environment check passed!")

    # 4. Initialize the SB3 agent (PPO)
    model = PPO("MlpPolicy", wrapped_env, verbose=1, tensorboard_log="./logs/sb3_blackjack_logs/")
    print("PPO agent initialized.")

    # 5. Train the agent
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print("✅ Training complete!")

    # 6. Save the trained model
    model.save("sb3_blackjack_ppo")
    print("Model saved to sb3_blackjack_ppo.zip")

    print("\n--- A game with the trained agent ---")
    obs, info = wrapped_env.reset()
    for _ in range(10): # Play 10 steps
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Action: {action}, Reward: {reward:.3f} (Original: {info['original_reward']:.2f}, RLlama: {info['rllama_reward']:.3f}), Done: {terminated or truncated}")
        if terminated or truncated:
            print("Game Over!")
            break

if __name__ == "__main__":
    main()