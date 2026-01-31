"""
EfficientNet-B0 Image Encoder for visual feature extraction.
Uses ONNX Runtime for optimized inference.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import base64
from typing import Union
import logging

logger = logging.getLogger(__name__)


class ImageEncoder:
    """EfficientNet-B0 based image encoder for feature extraction."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.transform = self._get_transform()
        self.embedding_dim = 1280  # EfficientNet-B0 output dimension
        
        logger.info(f"ImageEncoder initialized on device: {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load pre-trained EfficientNet-B0 model."""
        # Load pre-trained model
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Remove the classifier head to get embeddings
        model.classifier = nn.Identity()
        
        # Set to eval mode and move to device
        model.eval()
        model.to(self.device)
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def preprocess_image(self, image: Union[str, bytes, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Base64 string, bytes, or PIL Image
            
        Returns:
            Preprocessed tensor ready for model
        """
        # Handle base64 string
        if isinstance(image, str):
            image_bytes = base64.b64decode(image)
            image = Image.open(io.BytesIO(image_bytes))
        
        # Handle bytes
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0).to(self.device)
    
    def encode(self, image: Union[str, bytes, Image.Image]) -> np.ndarray:
        """
        Extract feature embedding from an image.
        
        Args:
            image: Base64 string, bytes, or PIL Image
            
        Returns:
            1D numpy array of shape (1280,) - the embedding vector
        """
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Extract features
        with torch.no_grad():
            embedding = self.model(tensor)
        
        # Convert to numpy and normalize
        embedding = embedding.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
        
        return embedding
    
    def encode_batch(self, images: list) -> np.ndarray:
        """
        Extract embeddings for multiple images.
        
        Args:
            images: List of base64 strings, bytes, or PIL Images
            
        Returns:
            2D numpy array of shape (N, 1280)
        """
        embeddings = []
        for image in images:
            try:
                embedding = self.encode(image)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to encode image: {e}")
                continue
        
        return np.array(embeddings)


# Singleton instance
_encoder_instance = None


def get_encoder() -> ImageEncoder:
    """Get singleton encoder instance."""
    global _encoder_instance
    if _encoder_instance is None:
        _encoder_instance = ImageEncoder()
    return _encoder_instance
