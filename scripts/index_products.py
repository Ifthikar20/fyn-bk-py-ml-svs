"""
Script to batch index products from the main Fynda database.

Usage:
    python scripts/index_products.py --api-url http://localhost:8000
"""
import asyncio
import httpx
import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import get_encoder
from app.services import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_products_from_api(api_url: str, limit: int = 1000) -> list:
    """Fetch products from the main Fynda API."""
    async with httpx.AsyncClient() as client:
        try:
            # Fetch trending/popular products
            response = await client.get(
                f"{api_url}/api/search/",
                params={"q": "trending", "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("deals", [])
        except Exception as e:
            logger.error(f"Failed to fetch products: {e}")
            return []


async def index_product_image(encoder, vector_store, product: dict) -> bool:
    """Index a single product by its image URL."""
    product_id = str(product.get("id", ""))
    image_url = product.get("image_url") or product.get("image", "")
    
    if not product_id or not image_url:
        return False
    
    try:
        # Fetch image
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=10)
            response.raise_for_status()
            image_data = response.content
        
        # Encode image
        embedding = encoder.encode(image_data)
        
        # Add to index
        metadata = {
            "title": product.get("title", ""),
            "price": float(product.get("price", 0) or 0),
            "image_url": image_url,
            "merchant": product.get("merchant_name") or product.get("source", ""),
            "category": product.get("category", ""),
        }
        
        success = vector_store.add_product(
            product_id=product_id,
            embedding=embedding,
            metadata=metadata
        )
        
        return success
    
    except Exception as e:
        logger.warning(f"Failed to index product {product_id}: {e}")
        return False


async def main(api_url: str, limit: int):
    """Main indexing function."""
    logger.info(f"Starting product indexing from {api_url}")
    
    # Initialize encoder and vector store
    encoder = get_encoder()
    vector_store = get_vector_store()
    
    logger.info(f"Current index size: {vector_store.index.ntotal}")
    
    # Fetch products
    logger.info("Fetching products from API...")
    products = await fetch_products_from_api(api_url, limit)
    logger.info(f"Fetched {len(products)} products")
    
    # Index each product
    indexed = 0
    skipped = 0
    failed = 0
    
    for i, product in enumerate(products):
        success = await index_product_image(encoder, vector_store, product)
        
        if success:
            indexed += 1
        elif product.get("id") and vector_store.id_to_idx.get(str(product["id"])):
            skipped += 1
        else:
            failed += 1
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i + 1}/{len(products)} (indexed: {indexed}, skipped: {skipped}, failed: {failed})")
    
    # Save index
    vector_store.save()
    
    logger.info(f"Indexing complete!")
    logger.info(f"  Indexed: {indexed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total in index: {vector_store.index.ntotal}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index products for visual search")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Main Fynda API URL")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum products to fetch")
    
    args = parser.parse_args()
    asyncio.run(main(args.api_url, args.limit))
