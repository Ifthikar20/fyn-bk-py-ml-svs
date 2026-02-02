"""
Search API endpoints for visual similarity search.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import logging
import httpx

from ..models import get_encoder
from ..services import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


# Request/Response Models
class VisualSearchRequest(BaseModel):
    """Request model for visual search."""
    image_base64: str = Field(..., description="Base64 encoded image data")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")


class SearchResultItem(BaseModel):
    """A single search result."""
    product_id: str
    similarity_score: float
    metadata: Dict[str, Any]


class VisualSearchResponse(BaseModel):
    """Response model for visual search."""
    success: bool
    query_time_ms: float
    results: List[SearchResultItem]


class IndexProductRequest(BaseModel):
    """Request model for indexing a product."""
    product_id: str = Field(..., description="Unique product identifier")
    image_url: Optional[str] = Field(None, description="URL of product image")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Product metadata")


class IndexProductResponse(BaseModel):
    """Response model for indexing."""
    success: bool
    product_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_stats: Dict[str, Any]


# Endpoints
@router.post("/visual-search", response_model=VisualSearchResponse)
async def visual_search(request: VisualSearchRequest):
    """
    Search for visually similar products.
    
    Send a base64 encoded image and receive similar products from the index.
    """
    start_time = time.time()
    
    try:
        # Get encoder and vector store
        encoder = get_encoder()
        vector_store = get_vector_store()
        
        # Encode query image
        embedding = encoder.encode(request.image_base64)
        
        # Search for similar products
        results = vector_store.search(embedding, top_k=request.top_k)
        
        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000
        
        return VisualSearchResponse(
            success=True,
            query_time_ms=round(query_time_ms, 2),
            results=[
                SearchResultItem(
                    product_id=r.product_id,
                    similarity_score=round(r.similarity_score, 4),
                    metadata=r.metadata
                )
                for r in results
            ]
        )
    
    except Exception as e:
        logger.error(f"Visual search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-product", response_model=IndexProductResponse)
async def index_product(request: IndexProductRequest):
    """
    Add a product to the search index.
    
    Provide either image_url or image_base64.
    """
    try:
        encoder = get_encoder()
        vector_store = get_vector_store()
        
        # Get image data
        image_data = None
        
        if request.image_base64:
            image_data = request.image_base64
        elif request.image_url:
            # Fetch image from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(request.image_url, timeout=10)
                response.raise_for_status()
                image_data = response.content
        else:
            raise HTTPException(status_code=400, detail="Provide either image_url or image_base64")
        
        # Encode image
        embedding = encoder.encode(image_data)
        
        # Add to index
        success = vector_store.add_product(
            product_id=request.product_id,
            embedding=embedding,
            metadata=request.metadata
        )
        
        if success:
            # Auto-save after adding
            vector_store.save()
            return IndexProductResponse(
                success=True,
                product_id=request.product_id,
                message="Product indexed successfully"
            )
        else:
            return IndexProductResponse(
                success=False,
                product_id=request.product_id,
                message="Product already exists in index"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index product failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with index statistics."""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            index_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-index")
async def save_index():
    """Manually save the index to disk."""
    try:
        vector_store = get_vector_store()
        vector_store.save()
        return {"success": True, "message": "Index saved successfully"}
    except Exception as e:
        logger.error(f"Save index failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ATTRIBUTE EXTRACTION ENDPOINTS
# ============================================================================

class ExtractAttributesRequest(BaseModel):
    """Request model for attribute extraction."""
    image_base64: str = Field(..., description="Base64 encoded image data")


class ColorInfo(BaseModel):
    """Color information."""
    hex: str
    name: str
    synonyms: List[str]
    rgb: List[int]


class ExtractAttributesResponse(BaseModel):
    """Response model for attribute extraction."""
    success: bool
    caption: str
    colors: Dict[str, ColorInfo]
    textures: List[str]
    category: str
    search_queries: List[str]
    processing_time_ms: float


@router.post("/extract-attributes", response_model=ExtractAttributesResponse)
async def extract_attributes(request: ExtractAttributesRequest):
    """
    Extract searchable attributes from an image.
    
    Returns caption, colors, textures, and generated search queries
    optimized for marketplace API searches.
    """
    start_time = time.time()
    
    try:
        from ..models import get_attribute_extractor
        
        # Get extractor
        extractor = get_attribute_extractor()
        
        # Extract attributes
        result = extractor.extract(request.image_base64)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Convert colors to response format
        colors_response = {}
        for key, color in result.colors.items():
            colors_response[key] = ColorInfo(
                hex=color.hex,
                name=color.name,
                synonyms=color.synonyms,
                rgb=list(color.rgb)
            )
        
        return ExtractAttributesResponse(
            success=True,
            caption=result.caption,
            colors=colors_response,
            textures=result.textures,
            category=result.category,
            search_queries=result.search_queries,
            processing_time_ms=round(processing_time_ms, 2)
        )
    
    except Exception as e:
        logger.error(f"Attribute extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

