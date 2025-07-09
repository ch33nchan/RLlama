from rllama.agents.base import BaseAgent
from rllama.agents.ppo import PPO
from rllama.agents.dqn import DQN
from rllama.agents.a2c import A2C
from rllama.agents.ddpg import DDPG
from rllama.agents.td3 import TD3
from rllama.agents.sac import SAC
from rllama.agents.mbpo import MBPO


def make_agent(algorithm: str, env, **kwargs) -> BaseAgent:
    """
    Create an agent based on the specified algorithm.
    
    Args:
        algorithm: Name of the algorithm
        env: Environment to interact with
        **kwargs: Additional arguments to pass to the agent constructor
        
    Returns:
        Agent instance
    """
    if algorithm.lower() == "ppo":
        return PPO(env=env, **kwargs)
    elif algorithm.lower() == "dqn":
        return DQN(env=env, **kwargs)
    elif algorithm.lower() == "a2c":
        return A2C(env=env, **kwargs)
    elif algorithm.lower() == "ddpg":
        return DDPG(env=env, **kwargs)
    elif algorithm.lower() == "td3":
        return TD3(env=env, **kwargs)
    elif algorithm.lower() == "sac":
        return SAC(env=env, **kwargs)
    elif algorithm.lower() == "mbpo":
        return MBPO(env=env, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")