# rllama/rewards/components/llm_rewards.py

from typing import Dict, Any
from rllama.rewards.base import BaseReward

# To avoid a hard dependency, we use a try-except block.
# This makes the library more modular.
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class HuggingFaceSentimentReward(BaseReward):
    """
    Uses a Hugging Face pipeline to reward positive sentiment in a response.
    """
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): The path to a sentiment-analysis model on the
                              Hugging Face Hub or a local path.
        """
        super().__init__()
        if pipeline is None:
            raise ImportError(
                "The 'transformers' library is required to use HuggingFaceSentimentReward. "
                "Please install it with: pip install rllama[trl]"
            )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", model=model_path, top_k=None
        )

    def calculate(self, context: Dict[str, Any]) -> float:
        response = context.get("response", "")
        if not isinstance(response, str) or not response:
            return 0.0
        
        try:
            # The pipeline returns a list of dictionaries for each label
            sentiments = self.sentiment_pipeline(
                response, truncation=True, max_length=512
            )[0]
            
            # Find the score for the 'POSITIVE' label
            positive_sentiment = next(
                (s for s in sentiments if s['label'] == 'POSITIVE'), None
            )
            
            if positive_sentiment:
                return positive_sentiment['score']
            return 0.0
        
        except Exception:
            # Handle cases where sentiment analysis might fail
            return 0.0