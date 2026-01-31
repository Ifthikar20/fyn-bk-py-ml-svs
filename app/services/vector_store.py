"""
FAISS-based vector store for similarity search.
"""
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProductMetadata:
    """Metadata associated with a product."""
    product_id: str
    title: str = ""
    price: float = 0.0
    image_url: str = ""
    merchant: str = ""
    category: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SearchResult:
    """A single search result."""
    product_id: str
    similarity_score: float
    metadata: Dict[str, Any]


class VectorStore:
    """FAISS-based vector store for product embeddings."""
    
    def __init__(self, dimension: int = 1280, index_path: str = "./data/faiss_index"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension (1280 for EfficientNet-B0)
            index_path: Path to store/load the index
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS index - using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Metadata storage: list of product metadata in same order as index
        self.metadata: List[ProductMetadata] = []
        
        # ID to index mapping
        self.id_to_idx: Dict[str, int] = {}
        
        # Try to load existing index
        self._load_if_exists()
        
        logger.info(f"VectorStore initialized with {self.index.ntotal} products")
    
    def _load_if_exists(self):
        """Load index and metadata if they exist."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.json"
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    self.metadata = [ProductMetadata(**m) for m in data["metadata"]]
                    self.id_to_idx = data["id_to_idx"]
                
                logger.info(f"Loaded existing index with {self.index.ntotal} products")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
    
    def save(self):
        """Save index and metadata to disk."""
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.json"
        
        faiss.write_index(self.index, str(index_file))
        
        with open(metadata_file, "w") as f:
            json.dump({
                "metadata": [m.to_dict() for m in self.metadata],
                "id_to_idx": self.id_to_idx
            }, f)
        
        logger.info(f"Saved index with {self.index.ntotal} products")
    
    def add_product(
        self,
        product_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a product embedding to the index.
        
        Args:
            product_id: Unique product identifier
            embedding: Normalized embedding vector (1280,)
            metadata: Optional product metadata
            
        Returns:
            True if added successfully
        """
        # Check if already exists
        if product_id in self.id_to_idx:
            logger.warning(f"Product {product_id} already exists, skipping")
            return False
        
        # Ensure embedding is 2D for FAISS
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        # Ensure float32
        embedding = embedding.astype(np.float32)
        
        # Add to index
        self.index.add(embedding)
        
        # Store metadata
        idx = self.index.ntotal - 1
        self.id_to_idx[product_id] = idx
        
        product_metadata = ProductMetadata(
            product_id=product_id,
            **(metadata or {})
        )
        self.metadata.append(product_metadata)
        
        return True
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar products.
        
        Args:
            query_embedding: Query embedding vector (1280,)
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of SearchResult objects
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure embedding is 2D for FAISS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Ensure float32
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for no result
                continue
            if score < threshold:
                continue
            
            metadata = self.metadata[idx]
            results.append(SearchResult(
                product_id=metadata.product_id,
                similarity_score=float(score),
                metadata=metadata.to_dict()
            ))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_products": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP"
        }


# Singleton instance
_store_instance = None


def get_vector_store(dimension: int = 1280, index_path: str = "./data/faiss_index") -> VectorStore:
    """Get singleton vector store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore(dimension=dimension, index_path=index_path)
    return _store_instance
