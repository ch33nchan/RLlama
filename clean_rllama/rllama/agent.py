# rllama/agent.py

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import os
import json
import yaml
from datetime import datetime
from collections import deque
import time

from .memory import EpisodicMemory, WorkingMemory, MemoryEntry, MemoryCompressor
from .rewards.engine import RewardEngine
from .rewards.normalization import RewardNormalizer, AdaptiveNormalizer

@dataclass
class AgentExperience:
    """A single agent experience sample"""
    state: Any
    action: Any
    next_state: Any
    reward: float
    done: bool
    info: Dict[str, Any] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.info is None:
            self.info = {}


class RLlamaAgent:
    """
    RLlama Agent class that handles interaction with environment, 
    manages rewards, and uses memory systems.
    """
    
    def __init__(self,
                 model: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 device: str = "cpu",
                 reward_config: Optional[Union[str, Dict[str, Any]]] = None,
                 memory_config: Optional[Dict[str, Any]] = None,
                 normalize_rewards: bool = True,
                 **kwargs):
        """
        Initialize the RLlama agent.
        
        Args:
            model: The model to use for policy decisions (optional).
            tokenizer: Tokenizer for model input/output processing (optional).
            device: Device to run the model on ('cpu', 'cuda', 'mps').
            reward_config: Path to reward config file or dict with reward config.
            memory_config: Memory configuration parameters.
            normalize_rewards: Whether to normalize rewards.
            **kwargs: Additional arguments.
        """
        # Core components
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Set up reward system
        self.reward_engine = None
        if reward_config:
            if isinstance(reward_config, str) and os.path.exists(reward_config):
                self.reward_engine = RewardEngine(reward_config)
            elif isinstance(reward_config, dict):
                # Create temporary config file to initialize reward engine
                import tempfile
                with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
                    yaml.dump(reward_config, f)
                    temp_path = f.name
                self.reward_engine = RewardEngine(temp_path)
                os.unlink(temp_path)
        
        # Set up memory systems
        memory_config = memory_config or {}
        self.use_memory = memory_config.get("use_memory", True)
        if self.use_memory:
            self.episodic_memory = EpisodicMemory(
                capacity=memory_config.get("episodic_capacity", 10000),
                importance_decay=memory_config.get("importance_decay", 0.99)
            )
            self.working_memory = WorkingMemory(
                max_size=memory_config.get("working_memory_size", 10),
                attention_temperature=memory_config.get("attention_temp", 1.0)
            )
            self.memory_compressor = MemoryCompressor(
                compression_ratio=memory_config.get("compression_ratio", 0.5)
            )
        
        # Reward normalization
        self.normalize_rewards = normalize_rewards
        if normalize_rewards:
            self.reward_normalizer = AdaptiveNormalizer(
                window_size=kwargs.get("reward_window_size", 100)
            )
        
        # Tracking and history
        self.total_steps = 0
        self.episode_count = 0
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_experiences = []
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Configuration
        self.generate_kwargs = kwargs.get("generate_kwargs", {})
        self.algorithm = kwargs.get("algorithm", "ppo")
        
        # Set up logging
        self.log_dir = kwargs.get("log_dir", "./rllama_logs")
        os.makedirs(self.log_dir, exist_ok=True)
    
    def reset(self):
        """Reset the agent for a new episode"""
        self.current_episode_steps = 0
        self.current_episode_reward = 0.0
        self.current_episode_experiences = []
        self.episode_count += 1
        
        # Clear working memory but keep episodic memory
        if self.use_memory:
            self.working_memory = WorkingMemory(
                max_size=self.working_memory.memory.maxlen,
                attention_temperature=self.working_memory.attention_temperature
            )
    
    def act(self, 
            state: Any, 
            deterministic: bool = False, 
            **kwargs) -> Any:
        """
        Generate an action based on the current state.
        
        Args:
            state: The current environment state.
            deterministic: Whether to act deterministically.
            **kwargs: Additional arguments for action generation.
            
        Returns:
            The selected action.
        """
        # Convert state to something usable if needed
        processed_state = self.preprocess_state(state)
        
        # Get state embedding for memory lookup
        state_embedding = self.get_state_embedding(processed_state)
        
        # Use memory systems if enabled
        if self.use_memory:
            # Retrieve relevant past experiences
            relevant_memories = self.episodic_memory.retrieve_relevant(state_embedding, k=3)
            
            # Update working memory
            for memory in relevant_memories:
                self.working_memory.add(memory.state)
            
            # Get context-enhanced state
            context_embedding = self.working_memory.get_context(state_embedding)
            
            # Use enhanced state for decision making
            action = self._generate_action(processed_state, context_embedding, deterministic, **kwargs)
        else:
            # Generate action without memory enhancement
            action = self._generate_action(processed_state, state_embedding, deterministic, **kwargs)
        
        self.current_episode_steps += 1
        self.total_steps += 1
        
        return action
    
    def observe(self, 
                state: Any, 
                action: Any, 
                reward: float, 
                next_state: Any, 
                done: bool, 
                info: Dict[str, Any] = None) -> None:
        """
        Process a new observation/experience.
        
        Args:
            state: The state in which the action was taken.
            action: The action that was taken.
            reward: The reward received.
            next_state: The resulting state.
            done: Whether the episode is done.
            info: Additional information.
        """
        # Record experience
        info = info or {}
        exp = AgentExperience(state, action, next_state, reward, done, info)
        self.current_episode_experiences.append(exp)
        
        # Track episode reward
        self.current_episode_reward += reward
        
        # Store in episodic memory if enabled
        if self.use_memory:
            state_embedding = self.get_state_embedding(self.preprocess_state(state))
            next_state_embedding = self.get_state_embedding(self.preprocess_state(next_state))
            
            self.episodic_memory.add(MemoryEntry(
                state=state_embedding,
                action=action,
                reward=reward,
                next_state=next_state_embedding,
                done=done,
                timestamp=int(time.time()),
                importance=abs(reward),
                metadata=info
            ))
        
        # End of episode cleanup
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_steps)
            
            # Log episode results
            self._log_episode()
            
            # Reset for next episode
            self.reset()
    
    def get_state_embedding(self, state: Any) -> torch.Tensor:
        """
        Get an embedding representation of the state.
        
        Args:
            state: The state to embed.
            
        Returns:
            Tensor representation of the state.
        """
        # If the model is a transformer model
        if self.model is not None and self.tokenizer is not None:
            if isinstance(state, str):
                # Tokenize the state
                inputs = self.tokenizer(state, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                with torch.no_grad():
                    outputs = self.model(
                        **inputs, 
                        output_hidden_states=True
                    )
                    
                    # Get the embeddings from the last hidden layer
                    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                    # Get mean of token embeddings
                    embedding = torch.mean(hidden_states, dim=1)
                    
                    return embedding
            
            elif isinstance(state, dict) and 'input_ids' in state:
                # State is already tokenized
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) 
                         for k, v in state.items()}
                
                with torch.no_grad():
                    outputs = self.model(
                        **inputs, 
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                    embedding = torch.mean(hidden_states, dim=1)
                    return embedding
                
        # Fallback for other types of states
        if isinstance(state, torch.Tensor):
            return state.to(self.device)
        elif isinstance(state, np.ndarray):
            return torch.tensor(state, device=self.device).float()
        
        # Last resort - convert to tensor if possible
        try:
            return torch.tensor(state, device=self.device).float()
        except:
            # If we can't convert to tensor, we'll need a custom embedding function
            # This should be overridden in subclasses for specific state types
            raise NotImplementedError(
                f"get_state_embedding not implemented for state type: {type(state)}"
            )
    
    def preprocess_state(self, state: Any) -> Any:
        """
        Preprocess state before action generation.
        Subclasses should override this for custom preprocessing.
        """
        return state
    
    def postprocess_action(self, action: Any) -> Any:
        """
        Postprocess the action before returning it.
        Subclasses should override this for custom postprocessing.
        """
        return action
    
    def _generate_action(self, 
                        state: Any, 
                        state_embedding: torch.Tensor, 
                        deterministic: bool = False, 
                        **kwargs) -> Any:
        """
        Internal method to generate an action.
        Subclasses should override this for custom action generation.
        """
        # This base implementation just returns a random action
        # Subclasses should implement actual model-based policy
        return {"action": "base_action_not_implemented"}
    
    def compute_reward(self, context: Dict[str, Any]) -> float:
        """
        Compute reward for a given context using the reward engine.
        
        Args:
            context: Context dictionary with information for reward calculation.
            
        Returns:
            The calculated reward value.
        """
        if self.reward_engine is None:
            # No reward engine, return a default 0 reward
            return 0.0
        
        raw_reward = self.reward_engine.compute_and_log(context)
        
        if self.normalize_rewards:
            return self.reward_normalizer.normalize(raw_reward)
        else:
            return raw_reward
    
    def _log_episode(self):
        """Log episode results"""
        # Create basic log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode_count,
            "total_steps": self.total_steps,
            "episode_steps": self.current_episode_steps,
            "episode_reward": self.current_episode_reward,
        }
        
        # Add rolling statistics if we have enough episodes
        if len(self.episode_rewards) > 0:
            log_entry["avg_reward_10"] = np.mean(self.episode_rewards[-10:])
            log_entry["avg_length_10"] = np.mean(self.episode_lengths[-10:])
            
        # Save log entry
        log_file = os.path.join(self.log_dir, f"episode_{self.episode_count}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Also append to a consolidated log file
        consolidated_log = os.path.join(self.log_dir, "episodes.jsonl")
        with open(consolidated_log, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def save(self, path: str):
        """
        Save the agent's state.
        
        Args:
            path: Directory path to save agent state.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save metadata
        metadata = {
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "timestamp": datetime.now().isoformat(),
            "version": "0.3.0"
        }
        
        with open(os.path.join(path, "agent_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save model if available
        if self.model is not None and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(os.path.join(path, "model"))
        
        # Save tokenizer if available
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
    
    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs):
        """
        Load an agent from saved state.
        
        Args:
            path: Directory path where agent state was saved.
            device: Device to load the model on.
            **kwargs: Additional arguments for agent initialization.
            
        Returns:
            The loaded agent.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load metadata
        metadata_path = os.path.join(path, "agent_metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"No agent metadata found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model and tokenizer if available
        model_path = os.path.join(path, "model")
        tokenizer_path = os.path.join(path, "tokenizer")
        
        model = None
        tokenizer = None
        
        if os.path.exists(model_path):
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Create agent instance
        agent = cls(model=model, tokenizer=tokenizer, device=device, **kwargs)
        
        # Restore agent state
        agent.total_steps = metadata.get("total_steps", 0)
        agent.episode_count = metadata.get("episode_count", 0)
        agent.episode_rewards = metadata.get("episode_rewards", [])
        agent.episode_lengths = metadata.get("episode_lengths", [])
        
        return agent