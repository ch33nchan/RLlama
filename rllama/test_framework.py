import os
import sys
import gymnasium as gym
import numpy as np
import rllama

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports... ", end="")
    try:
        from rllama.agents.dqn import DQN
        from rllama.agents.a2c import A2C
        from rllama.agents.ppo import PPO
        from rllama.agents.ddpg import DDPG
        from rllama.agents.td3 import TD3
        from rllama.agents.sac import SAC
        from rllama.agents.mbpo import MBPO
        print("SUCCESS")
        return True
    except ImportError as e:
        print(f"FAILED - {e}")
        return False

def test_make_env():
    """Test that environments can be created."""
    print("Testing environment creation... ", end="")
    try:
        env = rllama.make_env("CartPole-v1", seed=42)
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()
        print("SUCCESS")
        return True
    except Exception as e:
        print(f"FAILED - {e}")
        return False

def test_make_agent():
    """Test that agents can be created."""
    print("Testing agent creation... ", end="")
    try:
        env = rllama.make_env("CartPole-v1", seed=42)
        agent = rllama.make_agent("DQN", env=env)
        obs, _ = env.reset()
        action = agent.select_action(obs)
        assert isinstance(action, (int, np.int64, np.ndarray))
        env.close()
        print("SUCCESS")
        return True
    except Exception as e:
        print(f"FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("RLlama Framework Test Suite")
    print("==========================")
    
    all_pass = True
    all_pass &= test_imports()
    all_pass &= test_make_env()
    all_pass &= test_make_agent()
    
    if all_pass:
        print("\nAll tests passed! The framework is working correctly.")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()