from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    app_name: str = "FYNDA ML Services"
    debug: bool = False
    port: int = 8001
    
    # Model
    model_name: str = "efficientnet_b0"
    embedding_dim: int = 1280  # EfficientNet-B0 output dimension
    
    # FAISS
    faiss_index_path: str = "./data/faiss_index"
    
    # Search
    max_results: int = 20
    similarity_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        env_prefix = "ML_"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
