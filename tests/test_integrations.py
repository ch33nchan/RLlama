import unittest
import os
import gymnasium as gym

import rllama
from rllama.integrations import sb3, mujoco, lerobot, unity


class TestIntegrations(unittest.TestCase):
    """Tests for integrations with external libraries."""
    
    def test_sb3_import(self):
        """Test Stable Baselines3 integration imports."""
        self.assertTrue(hasattr(sb3, 'from_sb3'))
        self.assertTrue(hasattr(sb3, 'to_sb3'))
        
    def test_mujoco_import(self):
        """Test MuJoCo integration imports."""
        self.assertTrue(hasattr(mujoco, 'wrap_mujoco'))
        
    def test_lerobot_import(self):
        """Test LeRobot integration imports."""
        self.assertTrue(hasattr(lerobot, 'wrap_lerobot'))
        
    def test_unity_import(self):
        """Test Unity integration imports."""
        self.assertTrue(hasattr(unity, 'wrap_unity'))


if __name__ == "__main__":
    unittest.main()