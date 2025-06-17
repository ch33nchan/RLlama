# rllama/integration/trl_wrapper.py

from typing import List, Dict, Any
import torch
from ..engine import RewardEngine  # Use relative import from parent directory

class TRLRlamaRewardProcessor:
    def __init__(self, config_path: str):
        self.engine = RewardEngine(config_path=config_path)

    def compute_reward(
        self,
        responses: List[str],
        prompts: List[str] = None,
        infos: List[Dict[str, Any]] = None
    ) -> torch.Tensor:
        rewards = []
        for i, response in enumerate(responses):
            context = {
                "response": response,
                "prompt": prompts[i] if prompts else None,
                "info": infos[i] if infos else None,
            }
            final_reward = self.engine.compute_and_log(context)
            rewards.append(torch.tensor(final_reward, dtype=torch.float32))
            
        return torch.stack(rewards)