import unittest
import gymnasium as gym
import numpy as np
import torch

import rllama
from rllama.agents import PPO, DQN, SAC, DDPG, NPPO, NDQN, NSAC


class TestAgents(unittest.TestCase):
    """Tests for RL agents."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simple environment for testing
        self.env = gym.make("CartPole-v1")
        self.continuous_env = gym.make("Pendulum-v1")
        
    def test_ppo_init(self):
        """Test PPO initialization."""
        agent = PPO(env=self.env)
        self.assertIsInstance(agent, PPO)
        
    def test_dqn_init(self):
        """Test DQN initialization."""
        agent = DQN(env=self.env)
        self.assertIsInstance(agent, DQN)
        
    def test_nppo_init(self):
        """Test NPPO initialization."""
        agent = NPPO(env=self.env)
        self.assertIsInstance(agent, NPPO)
        
    def test_ndqn_init(self):
        """Test NDQN initialization."""
        agent = NDQN(env=self.env)
        self.assertIsInstance(agent, NDQN)
        
    def test_sac_init(self):
        """Test SAC initialization."""
        agent = SAC(env=self.continuous_env)
        self.assertIsInstance(agent, SAC)
        
    def test_nsac_init(self):
        """Test NSAC initialization."""
        agent = NSAC(env=self.continuous_env)
        self.assertIsInstance(agent, NSAC)
        
    def test_ddpg_init(self):
        """Test DDPG initialization."""
        agent = DDPG(env=self.continuous_env)
        self.assertIsInstance(agent, DDPG)
        
    def test_make_agent(self):
        """Test make_agent function."""
        agent = rllama.make_agent("PPO", env=self.env)
        self.assertIsInstance(agent, PPO)
        
        agent = rllama.make_agent("DQN", env=self.env)
        self.assertIsInstance(agent, DQN)
        
        agent = rllama.make_agent("NPPO", env=self.env)
        self.assertIsInstance(agent, NPPO)
        
        agent = rllama.make_agent("NDQN", env=self.env)
        self.assertIsInstance(agent, NDQN)


if __name__ == "__main__":
    unittest.main()