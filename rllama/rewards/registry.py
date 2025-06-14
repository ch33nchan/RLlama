# rllama/rewards/registry.py

from rllama.rewards.components.common import LengthReward, ConstantReward
from rllama.rewards.components.llm_rewards import HuggingFaceSentimentReward

# A dictionary that maps the string names from your YAML file to
# the actual Python classes for the reward components.
# To add a new reward, simply define the class and add it here.
REWARD_REGISTRY = {
    # Common Rewards
    "LengthReward": LengthReward,
    "ConstantReward": ConstantReward,

    # LLM-specific Rewards
    "HuggingFaceSentimentReward": HuggingFaceSentimentReward,
}