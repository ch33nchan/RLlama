import unittest
import gymnasium as gym
import numpy as np

import rllama
from rllama.environments import make_env
from rllama.environments.wrappers import NormalizeObservation, NormalizeReward, TimeLimit, FrameStack


class TestEnvironments(unittest.TestCase):
    """Tests for environment wrappers."""
    
    def test_make_env(self):
        """Test make_env function."""
        env = make_env("CartPole-v1")
        self.assertIsInstance(env, gym.Env)
        
    def test_normalize_observation(self):
        """Test NormalizeObservation wrapper."""
        env = make_env("CartPole-v1", normalize_obs=True)
        self.assertIsInstance(env, NormalizeObservation)
        
    def test_normalize_reward(self):
        """Test NormalizeReward wrapper."""
        env = make_env("CartPole-v1", normalize_reward=True)
        self.assertIsInstance(env, NormalizeReward)
        
    def test_time_limit(self):
        """Test TimeLimit wrapper."""
        env = make_env("CartPole-v1", time_limit=100)
        self.assertIsInstance(env, TimeLimit)
        
    def test_frame_stack(self):
        """Test FrameStack wrapper."""
        env = make_env("CartPole-v1", frame_stack=4)
        self.assertIsInstance(env, FrameStack)
        
    def test_multiple_wrappers(self):
        """Test multiple wrappers."""
        env = make_env(
            "CartPole-v1",
            normalize_obs=True,
            normalize_reward=True,
            time_limit=100,
            frame_stack=4
        )
        self.assertIsInstance(env, FrameStack)


if __name__ == "__main__":
    unittest.main()