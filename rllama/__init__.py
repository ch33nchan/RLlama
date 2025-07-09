# rllama/__init__.py
from typing import Any, Dict, Optional

import gymnasium as gym

from rllama.agents.dqn import DQN
from rllama.agents.a2c import A2C
from rllama.agents.ppo import PPO
from rllama.agents.ddpg import DDPG
from rllama.agents.td3 import TD3
from rllama.agents.sac import SAC
from rllama.agents.mbpo import MBPO
from rllama.core.experiment import Experiment

# Remove the problematic import:
# from rllama.agents.n_ppo import NPPO

def make_env(env_id: str, seed: Optional[int] = None, **kwargs) -> gym.Env:
    """Create and seed a Gymnasium environment."""
    env = gym.make(env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env

def make_agent(agent_type: str, env: gym.Env, **kwargs) -> Any:
    """Create an agent of the specified type."""
    agent_cls = {
        "DQN": DQN,
        "A2C": A2C,
        "PPO": PPO,
        "DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC,
        "MBPO": MBPO,
    }[agent_type]
    
    return agent_cls(env=env, **kwargs)

__version__ = "0.1.0"