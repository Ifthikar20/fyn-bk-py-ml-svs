"""
FYNDA ML Services - Visual Similarity Search API

A microservice for visual product similarity search using EfficientNet + FAISS.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from .config import get_settings
from .routers import search_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Visual similarity search API for Fynda marketplace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting {settings.app_name}...")
    
    # Pre-load the encoder model
    from .models import get_encoder
    from .services import get_vector_store
    
    logger.info("Loading EfficientNet encoder...")
    encoder = get_encoder()
    logger.info(f"Encoder loaded. Embedding dimension: {encoder.embedding_dim}")
    
    logger.info("Loading FAISS vector store...")
    vector_store = get_vector_store(
        dimension=encoder.embedding_dim,
        index_path=settings.faiss_index_path
    )
    logger.info(f"Vector store loaded. Total products: {vector_store.index.ntotal}")
    
    logger.info(f"{settings.app_name} ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    
    # Save index before shutdown
    from .services import get_vector_store
    try:
        vector_store = get_vector_store()
        vector_store.save()
        logger.info("Index saved successfully")
    except Exception as e:
        logger.error(f"Failed to save index: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs"
    }
