# rllama/integration/trl_wrapper.py

import torch
from typing import Dict, List, Any, Optional, Union
import numpy as np
from ..engine import RewardEngine

class TRLRlamaRewardProcessor:
    """
    Wrapper for integrating RLlama with HuggingFace's TRL library.
    Processes rewards for TRL-based reinforcement learning.
    """
    
    def __init__(self, 
                 reward_config_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 normalize: bool = True,
                 kl_coef: float = 0.1,
                 combine_rewards: bool = True):
        """
        Initialize the TRL reward processor.
        
        Args:
            reward_config_path: Path to the RLlama reward configuration.
            device: Device to process rewards on.
            normalize: Whether to normalize rewards.
            kl_coef: KL divergence coefficient for penalizing divergence from reference.
            combine_rewards: Whether to combine RLlama rewards with TRL rewards.
        """
        self.reward_engine = RewardEngine(reward_config_path)
        self.device = device
        self.normalize = normalize
        self.kl_coef = kl_coef
        self.combine_rewards = combine_rewards
        
        self.reward_stats = {
            "count": 0,
            "mean": 0.0,
            "var": 0.0,
            "min": float("inf"),
            "max": float("-inf")
        }
    
    def __call__(self, 
                 prompts: List[str], 
                 responses: List[str],
                 scores: Optional[torch.Tensor] = None,
                 logprobs: Optional[torch.Tensor] = None,
                 ref_logprobs: Optional[torch.Tensor] = None,
                 metadata: Optional[List[Dict[str, Any]]] = None) -> torch.Tensor:
        """
        Process rewards for TRL.
        
        Args:
            prompts: List of prompt strings.
            responses: List of response strings.
            scores: Optional rewards already calculated by TRL.
            logprobs: Log probabilities of the model.
            ref_logprobs: Log probabilities of the reference model.
            metadata: Additional metadata for reward calculation.
            
        Returns:
            Tensor of rewards for each response.
        """
        batch_size = len(prompts)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # Calculate rewards using RLlama
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # Create context for reward calculation
            context = {
                "prompt": prompt,
                "response": response,
                "info": metadata[i] if metadata and i < len(metadata) else {}
            }
            
            # Compute reward
            reward = self.reward_engine.compute_and_log(context)
            rewards[i] = torch.tensor(reward, device=self.device)
        
        # Add KL penalty if available
        if logprobs is not None and ref_logprobs is not None:
            kl_divergence = logprobs - ref_logprobs
            kl_penalty = -self.kl_coef * kl_divergence.mean(dim=-1)
            
            # Add to rewards
            if self.combine_rewards:
                rewards += kl_penalty
            else:
                # If not combining, KL penalty is the only component
                rewards = kl_penalty
        
        # Combine with TRL rewards if available and requested
        if scores is not None and self.combine_rewards:
            rewards = rewards + scores
        
        # Normalize rewards if requested
        if self.normalize:
            rewards = self._normalize_rewards(rewards)
        
        return rewards
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics"""
        # Update statistics
        rewards_np = rewards.detach().cpu().numpy()
        batch_size = len(rewards_np)
        
        self.reward_stats["count"] += batch_size
        old_mean = self.reward_stats["mean"]
        
        # Update mean and variance
        self.reward_stats["mean"] = old_mean + np.sum(rewards_np - old_mean) / self.reward_stats["count"]
        
        if self.reward_stats["count"] > 1:
            batch_var = np.var(rewards_np, ddof=1) if batch_size > 1 else 0
            old_s = self.reward_stats["var"] * (self.reward_stats["count"] - batch_size - 1)
            new_s = batch_var * (batch_size - 1)
            batch_mean_var = np.var([old_mean, self.reward_stats["mean"]]) * batch_size
            
            # Update running variance
            self.reward_stats["var"] = (old_s + new_s + batch_mean_var) / (self.reward_stats["count"] - 1)
        
        # Update min/max
        self.reward_stats["min"] = min(self.reward_stats["min"], np.min(rewards_np))
        self.reward_stats["max"] = max(self.reward_stats["max"], np.max(rewards_np))
        
        # Perform normalization
        if self.reward_stats["count"] > 1 and np.sqrt(self.reward_stats["var"]) > 1e-8:
            normalized_rewards = (rewards - self.reward_stats["mean"]) / np.sqrt(self.reward_stats["var"])
            return normalized_rewards
        
        # If insufficient data or variance is too small, just center the rewards
        return rewards - rewards.mean() + 0.1
    
    def reset_statistics(self):
        """Reset normalization statistics"""
        self.reward_stats = {
            "count": 0,
            "mean": 0.0,
            "var": 0.0,
            "min": float("inf"),
            "max": float("-inf")
        }