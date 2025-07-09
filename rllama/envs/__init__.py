"""
Environment wrappers for RLlama.
"""
try:
    from rllama.envs.lerobot import (
        LeRobotWrapper, 
        make_lerobot_env, 
        list_lerobot_envs
    )
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False