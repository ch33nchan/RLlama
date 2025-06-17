# rllama/engine.py

import yaml
from typing import Dict, Any
from .rewards.composer import RewardComposer
from .rewards.shaper import RewardShaper
from .rewards.registry import REWARD_REGISTRY
from .logger import RewardLogger
from datetime import datetime

class RewardEngine:
    """
    The central engine for processing rewards. It loads configurations,
    manages the reward composer and shaper, and orchestrates logging.
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.logger = RewardLogger(**config.get("logging", {}))

        reward_components_config = config.get('reward_components', [])
        reward_components = []
        for comp_config in reward_components_config:
            name = comp_config['name']
            params = comp_config.get('params', {})
            if name in REWARD_REGISTRY:
                reward_class = REWARD_REGISTRY[name]
                reward_components.append(reward_class(**params))
            else:
                raise ValueError(f"Reward component '{name}' not found in registry.")

        shaping_config = config.get('shaping_config', {})
        self.composer = RewardComposer(reward_components)
        self.shaper = RewardShaper(shaping_config)
        self.current_step = 0
        
        print("✅ RewardEngine initialized successfully.")

    def compute_and_log(self, context: Dict[str, Any]) -> float:
        context['step'] = self.current_step
        component_rewards = self.composer.calculate(context)
        final_reward = self.shaper.shape(component_rewards, step=self.current_step)

        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "context": {k: v for k, v in context.items() if k not in ['info', 'observation']},
            "component_rewards": {k: round(v, 4) for k, v in component_rewards.items()},
            "final_reward": round(final_reward, 4),
        }
        self.logger.log(trace_data)
        
        self.current_step += 1
        return final_reward