# rllama/rewards/vision_rewards.py

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import io
from .base import BaseReward

class VisualReasoningReward(BaseReward):
    """
    Reward component for visual reasoning tasks.
    Uses a visual language model to evaluate responses to image-based questions.
    """
    
    def __init__(self, 
                 model_name: str = "openai/clip-vit-large-patch14",
                 embedding_similarity_weight: float = 0.5,
                 answer_match_weight: float = 0.5,
                 use_cached_embeddings: bool = True):
        """
        Initialize the visual reasoning reward.
        
        Args:
            model_name: Name of the visual model to use.
            embedding_similarity_weight: Weight for embedding similarity component.
            answer_match_weight: Weight for answer match component.
            use_cached_embeddings: Whether to cache embeddings for efficiency.
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_similarity_weight = embedding_similarity_weight
        self.answer_match_weight = answer_match_weight
        self.use_cached_embeddings = use_cached_embeddings
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.embedding_cache = {}
    
    def _load_model(self):
        """Load the CLIP model for visual reasoning"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            return True
        except Exception as e:
            print(f"Error loading visual reasoning model: {e}")
            return False
    
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate visual reasoning reward based on image and response.
        
        Args:
            context: Dict containing:
                - image: PIL Image, file path, or raw bytes
                - query: Question about the image
                - response: Model's response 
                - reference_answer: (optional) Ground truth answer
                
        Returns:
            Reward score based on visual reasoning quality.
        """
        # Ensure model is loaded
        if self.model is None:
            if not self._load_model():
                return 0.0  # Return zero if model can't be loaded
        
        # Extract components from context
        image = context.get("image")
        query = context.get("query", "")
        response = context.get("response", "")
        reference_answer = context.get("reference_answer")
        
        # Convert image to PIL Image if needed
        pil_image = self._ensure_pil_image(image)
        if pil_image is None:
            return 0.0  # Invalid image
        
        # Calculate embedding similarity between image and response
        similarity_score = self._calculate_image_text_similarity(pil_image, query, response)
        
        # Calculate additional reward if reference answer is available
        answer_match_score = 0.0
        if reference_answer:
            answer_match_score = self._calculate_answer_similarity(response, reference_answer)
        
        # Calculate final reward as weighted combination
        final_reward = (
            self.embedding_similarity_weight * similarity_score +
            self.answer_match_weight * answer_match_score
        )
        
        return final_reward
    
    def _ensure_pil_image(self, image) -> Optional[Image.Image]:
        """Convert image to PIL Image if it's not already"""
        if image is None:
            return None
            
        if isinstance(image, Image.Image):
            return image
            
        if isinstance(image, str):
            try:
                return Image.open(image)
            except:
                return None
                
        if isinstance(image, bytes):
            try:
                return Image.open(io.BytesIO(image))
            except:
                return None
                
        if isinstance(image, np.ndarray):
            try:
                return Image.fromarray(image)
            except:
                return None
                
        return None
    
    def _calculate_image_text_similarity(self, image: Image.Image, query: str, response: str) -> float:
        """Calculate similarity between image and text response using CLIP"""
        # Create a combined query and response text
        combined_text = f"Question: {query} Answer: {response}"
        
        # Generate cache key for this image
        cache_key = hash(image.tobytes()) if self.use_cached_embeddings else None
        image_embedding = None
        
        # Try to get cached embedding
        if cache_key and cache_key in self.embedding_cache:
            image_embedding = self.embedding_cache[cache_key]
        
        # Process image and text through CLIP
        with torch.no_grad():
            # Get image embedding if not cached
            if image_embedding is None:
                inputs = self.processor(
                    text=[combined_text],
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to same device as model
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Get normalized embeddings
                image_embedding = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embedding = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                
                # Cache the image embedding
                if cache_key and self.use_cached_embeddings:
                    self.embedding_cache[cache_key] = image_embedding
            else:
                # Process only the text
                inputs = self.processor(
                    text=[combined_text],
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to same device as model
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                text_features = self.model.get_text_features(**inputs)
                text_embedding = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(image_embedding, text_embedding).item()
            
            # Normalize to [0, 1] range
            normalized_similarity = (similarity + 1) / 2
            
            return normalized_similarity
    
    def _calculate_answer_similarity(self, response: str, reference: str) -> float:
        """Calculate similarity between response and reference answer"""
        from nltk.translate.bleu_score import sentence_bleu
        import re
        
        # Simple text normalization
        def normalize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        
        response_norm = normalize(response)
        reference_norm = normalize(reference)
        
        # Tokenize
        response_tokens = response_norm.split()
        reference_tokens = reference_norm.split()
        
        # If either is empty, no match
        if not response_tokens or not reference_tokens:
            return 0.0
        
        # Calculate BLEU score (sentence-level)
        try:
            bleu_score = sentence_bleu([reference_tokens], response_tokens)
            return bleu_score
        except:
            # Fallback to token overlap
            response_set = set(response_tokens)
            reference_set = set(reference_tokens)
            
            if not reference_set:
                return 0.0
                
            overlap = len(response_set.intersection(reference_set))
            return overlap / len(reference_set)
    
    def reset(self):
        """Reset internal state between episodes"""
        # Clear embedding cache if it's too large
        if len(self.embedding_cache) > 1000:
            self.embedding_cache = {}


class VideoReasoningReward(BaseReward):
    """
    Reward component for video understanding tasks.
    Evaluates responses to questions about video content.
    """
    
    def __init__(self,
                 frame_sampling_rate: int = 5,
                 temporal_coherence_weight: float = 0.3,
                 visual_match_weight: float = 0.7,
                 max_frames: int = 10):
        """
        Initialize the video reasoning reward.
        
        Args:
            frame_sampling_rate: Sample every n frames from the video.
            temporal_coherence_weight: Weight for temporal coherence score.
            visual_match_weight: Weight for visual match score.
            max_frames: Maximum number of frames to process.
        """
        super().__init__()
        self.frame_sampling_rate = frame_sampling_rate
        self.temporal_coherence_weight = temporal_coherence_weight
        self.visual_match_weight = visual_match_weight
        self.max_frames = max_frames
        
        # Initialize visual reasoning component for frame-level processing
        self.visual_reasoner = VisualReasoningReward()
    
    def calculate(self, context: Dict[str, Any]) -> float:
        """
        Calculate video reasoning reward based on video and response.
        
        Args:
            context: Dict containing:
                - video_frames: List of frames (PIL Images or numpy arrays)
                - video_path: Alternatively, path to video file
                - query: Question about the video
                - response: Model's response
                - reference_answer: (optional) Ground truth answer
                
        Returns:
            Reward score based on video reasoning quality.
        """
        # Extract components from context
        frames = context.get("video_frames")
        video_path = context.get("video_path")
        query = context.get("query", "")
        response = context.get("response", "")
        reference_answer = context.get("reference_answer")
        
        # Get frames from video file if frames not provided directly
        if frames is None and video_path:
            frames = self._extract_frames(video_path)
        
        if not frames:
            return 0.0  # Invalid video input
        
        # Sample frames to avoid processing too many
        sampled_frames = self._sample_frames(frames)
        
        # Calculate visual match for each frame
        frame_scores = []
        for frame in sampled_frames:
            # Create a context for the visual reasoner
            frame_context = {
                "image": frame,
                "query": query,
                "response": response,
                "reference_answer": reference_answer
            }
            
            # Calculate visual reward for this frame
            frame_score = self.visual_reasoner.calculate(frame_context)
            frame_scores.append(frame_score)
        
        # Calculate average visual match score
        visual_match_score = np.mean(frame_scores) if frame_scores else 0.0
        
        # Calculate temporal coherence (consistency across frames)
        temporal_coherence = self._calculate_temporal_coherence(frame_scores)
        
        # Calculate final reward as weighted combination
        final_reward = (
            self.visual_match_weight * visual_match_score +
            self.temporal_coherence_weight * temporal_coherence
        )
        
        return final_reward
    
    def _extract_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from a video file"""
        try:
            import cv2
            
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return []
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % self.frame_sampling_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)
                    
                    if len(frames) >= self.max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []
    
    def _sample_frames(self, frames: List) -> List:
        """Sample frames to limit processing"""
        if len(frames) <= self.max_frames:
            return frames
            
        # Sample evenly across the video
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
        return [frames[i] for i in indices]
    
    def _calculate_temporal_coherence(self, frame_scores: List[float]) -> float:
        """
        Calculate temporal coherence as the consistency of scores across frames.
        Higher coherence means more consistent understanding across the video.
        """
        if len(frame_scores) <= 1:
            return 1.0  # Perfect coherence with only one frame
            
        # Calculate variance of scores (lower variance = higher coherence)
        variance = np.var(frame_scores)
        
        # Convert variance to coherence score (inverse relationship)
        coherence = 1.0 / (1.0 + 5.0 * variance)  # Scale factor of 5 for sensitivity
        
        return min(1.0, coherence)  # Cap at 1.0
    
    def reset(self):
        """Reset internal state between episodes"""
        self.visual_reasoner.reset()