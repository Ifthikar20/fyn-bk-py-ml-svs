"""
Fashion-CLIP Integration Module for Fynda ML Service.

This module loads a fine-tuned CLIP model (trained on fashion data via Colab)
and provides structured attribute extraction for uploaded images.

Usage:
    1. Train the model using notebooks/train_fashion_clip.py in Google Colab
    2. Download fashion_clip_fynda.pt + config.json from Google Drive
    3. Place them in FYNDA_ML_Services/models/fashion_clip/
    4. This module auto-loads the model at startup

The model replaces EfficientNet's generic captioning with fashion-aware
understanding: "purple striped men's jacket" instead of "makeup".
"""

import torch
import open_clip
from PIL import Image
from typing import Dict, List, Optional, Tuple
import json
import os
import logging
import time

logger = logging.getLogger(__name__)

# ============================================================================
# Fashion vocabularies for CLIP zero-shot classification
# ============================================================================

FASHION_CATEGORIES = [
    # Tops
    "t-shirt", "shirt", "blouse", "polo shirt", "tank top", "crop top",
    "henley", "button-down shirt", "flannel shirt",
    # Outerwear
    "jacket", "blazer", "coat", "hoodie", "windbreaker", "parka",
    "leather jacket", "denim jacket", "bomber jacket", "cardigan",
    "puffer jacket", "trench coat", "vest",
    # Bottoms
    "jeans", "pants", "trousers", "shorts", "skirt", "leggings",
    "sweatpants", "chinos", "cargo pants",
    # Dresses
    "dress", "maxi dress", "mini dress", "sundress", "cocktail dress",
    "evening gown", "wrap dress",
    # Sweaters
    "sweater", "pullover", "turtleneck", "crewneck sweater",
    # Footwear
    "sneakers", "boots", "sandals", "heels", "loafers", "flats",
    "running shoes", "dress shoes", "ankle boots", "slides",
    # Bags
    "handbag", "backpack", "tote bag", "crossbody bag", "clutch",
    "messenger bag", "duffel bag",
    # Accessories
    "hat", "cap", "beanie", "scarf", "belt", "watch",
    "sunglasses", "necklace", "bracelet", "earrings", "ring",
    # Activewear
    "sports bra", "yoga pants", "athletic shorts", "track jacket",
    "swimsuit", "bikini",
]

FASHION_COLORS = [
    "black", "white", "gray", "navy", "blue", "light blue", "sky blue",
    "red", "burgundy", "maroon", "crimson",
    "green", "olive", "forest green", "mint", "teal",
    "yellow", "gold", "mustard",
    "orange", "coral", "rust",
    "purple", "lavender", "plum", "violet",
    "pink", "hot pink", "blush", "rose",
    "brown", "tan", "beige", "camel", "cream",
    "silver", "charcoal", "ivory",
    "multicolor",
]

FASHION_PATTERNS = [
    "solid", "striped", "plaid", "checkered", "floral",
    "polka dot", "geometric", "abstract", "paisley",
    "animal print", "leopard print", "zebra print",
    "camouflage", "tie-dye", "color block",
    "embroidered", "sequined", "metallic",
]

FASHION_MATERIALS = [
    "cotton", "polyester", "silk", "linen", "wool",
    "denim", "leather", "suede", "velvet", "satin",
    "nylon", "cashmere", "lace", "mesh", "knit",
    "fleece", "chiffon", "tweed", "corduroy",
]

FASHION_STYLES = [
    "casual", "formal", "sporty", "bohemian", "vintage",
    "minimalist", "streetwear", "preppy", "punk", "romantic",
    "classic", "modern", "elegant", "edgy", "oversized",
    "fitted", "slim fit", "relaxed fit",
]

FASHION_GENDERS = [
    "men's", "women's", "unisex",
]


class FashionCLIP:
    """
    Fashion-aware CLIP model for structured attribute extraction.
    
    Uses zero-shot classification with fashion-specific vocabularies
    to extract garment type, colors, patterns, materials, and style.
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize FashionCLIP.
        
        Args:
            model_dir: Path to directory containing fashion_clip_fynda.pt + config.json.
                        If None, uses the default models/fashion_clip/ directory.
                        If no fine-tuned model found, falls back to base CLIP.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.is_fine_tuned = False
        
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'models', 'fashion_clip'
            )
        
        self._load_model(model_dir)
        self._precompute_text_features()
    
    def _load_model(self, model_dir: str):
        """Load the CLIP model (fine-tuned or base)."""
        # Read architecture from config.json (v4 uses ViT-B-16, v3 used ViT-B-32)
        config_path = os.path.join(model_dir, 'config.json')
        model_name = 'ViT-B-32'  # default fallback
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_name = config.get('model_name', 'ViT-B-32')
                version = config.get('version', 'unknown')
                logger.info(f"📋 Config: {model_name} (version: {version})")
            except Exception as e:
                logger.warning(f"Could not read config.json: {e}. Using default {model_name}")
        
        t0 = time.time()
        
        # Try to load fine-tuned model
        weights_path = os.path.join(model_dir, 'fashion_clip_fynda.pt')
        if os.path.exists(weights_path):
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained='openai', device=self.device
                )
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_fine_tuned = True
                logger.info(f"✅ Loaded fine-tuned Fashion-CLIP ({model_name}) from {weights_path}")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model: {e}. Falling back to base CLIP.")
                self._load_base_model(model_name)
        else:
            logger.info(f"No fine-tuned model at {weights_path}. Using base CLIP.")
            self._load_base_model(model_name)
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        logger.info(f"Model loaded in {(time.time()-t0)*1000:.0f}ms ({self.device})")
    
    def _load_base_model(self, model_name: str):
        """Load base OpenAI CLIP model."""
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained='openai', device=self.device
        )
        self.model.eval()
        self.is_fine_tuned = False
    
    def _precompute_text_features(self):
        """
        Pre-compute text embeddings for all fashion vocabularies.
        This makes inference fast — we only need to encode the image at runtime.
        """
        t0 = time.time()
        
        with torch.no_grad():
            # Category text features
            cat_texts = [f"a photo of a {c}" for c in FASHION_CATEGORIES]
            self._category_features = self._encode_texts(cat_texts)
            
            # Color text features  
            color_texts = [f"a {c} colored item" for c in FASHION_COLORS]
            self._color_features = self._encode_texts(color_texts)
            
            # Pattern text features
            pattern_texts = [f"a {p} pattern" for p in FASHION_PATTERNS]
            self._pattern_features = self._encode_texts(pattern_texts)
            
            # Material text features
            material_texts = [f"made of {m}" for m in FASHION_MATERIALS]
            self._material_features = self._encode_texts(material_texts)
            
            # Style text features
            style_texts = [f"a {s} style item" for s in FASHION_STYLES]
            self._style_features = self._encode_texts(style_texts)
            
            # Gender text features
            gender_texts = [f"a {g} clothing item" for g in FASHION_GENDERS]
            self._gender_features = self._encode_texts(gender_texts)
        
        logger.info(f"Pre-computed text features in {(time.time()-t0)*1000:.0f}ms")
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts into normalized feature vectors."""
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
        return features
    
    def _classify(
        self, 
        image_features: torch.Tensor, 
        text_features: torch.Tensor, 
        labels: List[str], 
        top_k: int = 3,
        threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Classify image against a set of labels using cosine similarity.
        
        Returns top_k labels with scores above threshold.
        """
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        scores, indices = similarity[0].topk(min(top_k, len(labels)))
        
        results = []
        for score, idx in zip(scores, indices):
            if score.item() >= threshold:
                results.append((labels[idx], round(score.item(), 3)))
        
        return results
    
    def extract_attributes(self, image: Image.Image) -> Dict:
        """
        Extract comprehensive fashion attributes from an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dict with keys: category, colors, pattern, material, style, 
                            gender, confidence, search_queries
        """
        t0 = time.time()
        
        # Preprocess and encode image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Classify against all vocabularies
        categories = self._classify(image_features, self._category_features, FASHION_CATEGORIES, top_k=3)
        colors = self._classify(image_features, self._color_features, FASHION_COLORS, top_k=3, threshold=0.05)
        patterns = self._classify(image_features, self._pattern_features, FASHION_PATTERNS, top_k=2)
        materials = self._classify(image_features, self._material_features, FASHION_MATERIALS, top_k=2, threshold=0.05)
        styles = self._classify(image_features, self._style_features, FASHION_STYLES, top_k=2, threshold=0.05)
        genders = self._classify(image_features, self._gender_features, FASHION_GENDERS, top_k=1)
        
        # Build result
        primary_category = categories[0][0] if categories else "clothing"
        primary_color = colors[0][0] if colors else ""
        secondary_colors = [c[0] for c in colors[1:] if c[1] > 0.1]
        primary_pattern = patterns[0][0] if patterns and patterns[0][0] != "solid" else ""
        primary_material = materials[0][0] if materials and materials[0][1] > 0.15 else ""
        primary_style = styles[0][0] if styles and styles[0][1] > 0.15 else ""
        gender = genders[0][0] if genders else ""
        
        # Generate search queries
        search_queries = self._generate_search_queries(
            category=primary_category,
            primary_color=primary_color,
            secondary_colors=secondary_colors,
            pattern=primary_pattern,
            material=primary_material,
            style=primary_style,
            gender=gender,
        )
        
        # Build caption
        caption_parts = []
        if primary_color:
            caption_parts.append(primary_color)
        if primary_pattern and primary_pattern != "solid":
            caption_parts.append(primary_pattern)
        if primary_material:
            caption_parts.append(primary_material)
        caption_parts.append(primary_category)
        caption = " ".join(caption_parts)
        
        elapsed = (time.time() - t0) * 1000
        logger.info(f"Fashion-CLIP extraction: {elapsed:.0f}ms → {caption}")
        
        return {
            "caption": caption,
            "category": primary_category,
            "colors": {
                "primary": primary_color,
                "secondary": secondary_colors,
                "all_detected": [(c[0], c[1]) for c in colors],
            },
            "pattern": primary_pattern or "solid",
            "material": primary_material,
            "style": primary_style,
            "gender": gender,
            "confidence": categories[0][1] if categories else 0.0,
            "search_queries": search_queries,
            "model": "fashion-clip-finetuned" if self.is_fine_tuned else "clip-base",
            "inference_ms": round(elapsed, 0),
        }
    
    def _generate_search_queries(
        self,
        category: str,
        primary_color: str,
        secondary_colors: List[str],
        pattern: str,
        material: str,
        style: str,
        gender: str,
    ) -> List[str]:
        """
        Generate optimized search queries for product search APIs.
        
        Produces multiple query variations to maximize result relevance:
        1. Full descriptive query
        2. Color + category (most common search pattern)
        3. Multi-color variant
        4. Pattern/material focused
        5. Gender-specific
        """
        queries = []
        
        # Query 1: Full descriptive — "purple striped men's jacket"
        parts = []
        if primary_color:
            parts.append(primary_color)
        if pattern:
            parts.append(pattern)
        if gender:
            parts.append(gender)
        parts.append(category)
        queries.append(" ".join(parts))
        
        # Query 2: Color + category — "purple jacket"
        if primary_color:
            queries.append(f"{primary_color} {category}")
        
        # Query 3: Multi-color — "purple and blue jacket"
        if primary_color and secondary_colors:
            color_str = f"{primary_color} and {secondary_colors[0]}"
            queries.append(f"{color_str} {category}")
        
        # Query 4: Material focused — "leather jacket"
        if material:
            queries.append(f"{material} {category}")
        
        # Query 5: Style focused — "casual jacket"
        if style:
            queries.append(f"{style} {category}")
        
        # Query 6: Pattern focused — "striped jacket"
        if pattern:
            queries.append(f"{pattern} {category}")
        
        # Query 7: Just category as fallback
        if category not in queries:
            queries.append(category)
        
        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        return unique_queries[:5]
    def rerank_by_text(
        self,
        query: str,
        image_urls: List[str],
        product_ids: List[str],
    ) -> List[Dict]:
        """
        Re-rank products by CLIP text-image similarity.
        
        Downloads product images, encodes them with CLIP, and scores
        against the text query. Returns products sorted by visual relevance.
        
        Args:
            query: Text search query (e.g., "blue tote bag for men")
            image_urls: List of product image URLs
            product_ids: Corresponding product IDs
            
        Returns:
            List of dicts with product_id and similarity_score, sorted desc.
        """
        import io
        import requests as req
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        t0 = time.time()
        
        # Step 1: Encode the text query
        with torch.no_grad():
            text_features = self._encode_texts([f"a photo of {query}"])
        
        # Step 2: Download and encode images in parallel
        def download_image(url: str) -> Optional[Image.Image]:
            try:
                resp = req.get(url, timeout=3, headers={
                    "User-Agent": "Mozilla/5.0"
                })
                if resp.status_code == 200:
                    return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                pass
            return None
        
        # Download all images in parallel
        images = [None] * len(image_urls)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {
                executor.submit(download_image, url): i
                for i, url in enumerate(image_urls)
            }
            for future in as_completed(future_to_idx, timeout=5):
                idx = future_to_idx[future]
                try:
                    images[idx] = future.result()
                except Exception:
                    pass
        
        download_ms = (time.time() - t0) * 1000
        
        # Step 3: Encode all successfully downloaded images
        results = []
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        
        if valid_indices:
            with torch.no_grad():
                image_tensors = torch.stack([
                    self.preprocess(images[i]) for i in valid_indices
                ]).to(self.device)
                
                image_features = self.model.encode_image(image_tensors)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Cosine similarity (text_features is 1×D, image_features is N×D)
                similarities = (text_features @ image_features.T).squeeze(0)
            
            for j, idx in enumerate(valid_indices):
                results.append({
                    "product_id": product_ids[idx],
                    "similarity_score": round(similarities[j].item(), 4),
                })
        
        # Add failed downloads with score 0
        scored_ids = {r["product_id"] for r in results}
        for pid in product_ids:
            if pid not in scored_ids:
                results.append({
                    "product_id": pid,
                    "similarity_score": 0.0,
                })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        encode_ms = (time.time() - t0) * 1000 - download_ms
        total_ms = (time.time() - t0) * 1000
        logger.info(
            f"CLIP re-rank: {len(valid_indices)}/{len(image_urls)} images scored "
            f"in {total_ms:.0f}ms (download: {download_ms:.0f}ms, encode: {encode_ms:.0f}ms) "
            f"for query: '{query}'"
        )
        
        return results


# ============================================================================
# Factory function — matches existing AttributeExtractor interface
# ============================================================================

_fashion_clip_instance: Optional[FashionCLIP] = None

def get_fashion_clip(model_dir: str = None) -> FashionCLIP:
    """Get or create the FashionCLIP singleton."""
    global _fashion_clip_instance
    if _fashion_clip_instance is None:
        _fashion_clip_instance = FashionCLIP(model_dir=model_dir)
    return _fashion_clip_instance
