"""
Base abstract class for Vision Language Models
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
from torch import nn
from PIL import Image


class BaseVLM(ABC, nn.Module):
    """Abstract base class for Vision Language Models"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the pre-trained model"""
        pass
    
    @abstractmethod
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocess image for model input"""
        pass
    
    @abstractmethod
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess text for model input"""
        pass
    
    @abstractmethod
    def generate_response(
        self, 
        image: Union[str, Image.Image], 
        question: str,
        max_length: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """Generate response given image and question"""
        pass
    
    @abstractmethod
    def get_attention_maps(
        self, 
        image: Union[str, Image.Image], 
        question: str
    ) -> Optional[torch.Tensor]:
        """Get attention maps for interpretability analysis"""
        pass
    
    def batch_generate(
        self,
        images: List[Union[str, Image.Image]],
        questions: List[str],
        **kwargs
    ) -> List[str]:
        """Generate responses for a batch of image-question pairs"""
        responses = []
        for image, question in zip(images, questions):
            response = self.generate_response(image, question, **kwargs)
            responses.append(response)
        return responses
    
    def evaluate_on_dataset(
        self,
        dataset: Any,
        evaluator: Any
    ) -> Dict[str, float]:
        """Evaluate model on a given dataset"""
        results = []
        for batch in dataset:
            if isinstance(batch, dict):
                image = batch.get('image')
                question = batch.get('question', '')
                ground_truth = batch.get('answer', '')
                
                prediction = self.generate_response(image, question)
                results.append({
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'image': image,
                    'question': question
                })
        
        return evaluator.evaluate(results) 