"""
Attribute Extractor for visual search enhancement.
Extracts color, texture, and generates captions using BLIP.
"""
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import io
import base64
import logging
from dataclasses import dataclass, asdict
from functools import lru_cache
import colorsys

logger = logging.getLogger(__name__)


# ============================================================================
# COLOR MAPPING - Fashion-relevant color names with synonyms
# ============================================================================

COLOR_MAP: Dict[str, Dict] = {
    # Reds
    "red": {"rgb": (255, 0, 0), "synonyms": ["crimson", "scarlet", "cherry", "ruby"]},
    "crimson": {"rgb": (220, 20, 60), "synonyms": ["red", "dark red", "deep red"]},
    "maroon": {"rgb": (128, 0, 0), "synonyms": ["burgundy", "wine", "dark red"]},
    "coral": {"rgb": (255, 127, 80), "synonyms": ["salmon", "peach", "light red"]},
    
    # Oranges
    "orange": {"rgb": (255, 165, 0), "synonyms": ["tangerine", "rust", "amber"]},
    "rust": {"rgb": (183, 65, 14), "synonyms": ["burnt orange", "terracotta", "copper"]},
    
    # Yellows
    "yellow": {"rgb": (255, 255, 0), "synonyms": ["golden", "lemon", "canary"]},
    "gold": {"rgb": (255, 215, 0), "synonyms": ["golden", "mustard", "amber"]},
    "mustard": {"rgb": (255, 219, 88), "synonyms": ["gold", "ochre", "honey"]},
    
    # Greens
    "green": {"rgb": (0, 128, 0), "synonyms": ["forest", "emerald", "jade"]},
    "olive": {"rgb": (128, 128, 0), "synonyms": ["army green", "khaki", "moss"]},
    "teal": {"rgb": (0, 128, 128), "synonyms": ["turquoise", "cyan", "aqua"]},
    "mint": {"rgb": (152, 255, 152), "synonyms": ["seafoam", "light green", "sage"]},
    "sage": {"rgb": (188, 184, 138), "synonyms": ["muted green", "moss", "dusty green"]},
    
    # Blues
    "blue": {"rgb": (0, 0, 255), "synonyms": ["royal", "cobalt", "azure"]},
    "navy": {"rgb": (0, 0, 128), "synonyms": ["dark blue", "midnight", "marine"]},
    "sky blue": {"rgb": (135, 206, 235), "synonyms": ["light blue", "baby blue", "powder blue"]},
    "teal blue": {"rgb": (54, 117, 136), "synonyms": ["ocean", "petrol", "peacock"]},
    "denim": {"rgb": (21, 96, 189), "synonyms": ["jean blue", "indigo", "chambray"]},
    
    # Purples
    "purple": {"rgb": (128, 0, 128), "synonyms": ["violet", "plum", "grape"]},
    "lavender": {"rgb": (230, 230, 250), "synonyms": ["lilac", "light purple", "mauve"]},
    "plum": {"rgb": (142, 69, 133), "synonyms": ["eggplant", "dark purple", "aubergine"]},
    
    # Pinks
    "pink": {"rgb": (255, 192, 203), "synonyms": ["rose", "blush", "salmon"]},
    "hot pink": {"rgb": (255, 105, 180), "synonyms": ["magenta", "fuchsia", "bright pink"]},
    "blush": {"rgb": (255, 111, 105), "synonyms": ["dusty rose", "mauve", "nude pink"]},
    
    # Browns
    "brown": {"rgb": (139, 69, 19), "synonyms": ["chocolate", "coffee", "espresso"]},
    "tan": {"rgb": (210, 180, 140), "synonyms": ["camel", "khaki", "sand"]},
    "beige": {"rgb": (245, 245, 220), "synonyms": ["cream", "nude", "ecru"]},
    "camel": {"rgb": (193, 154, 107), "synonyms": ["tan", "caramel", "toffee"]},
    
    # Neutrals
    "black": {"rgb": (0, 0, 0), "synonyms": ["jet", "onyx", "charcoal"]},
    "white": {"rgb": (255, 255, 255), "synonyms": ["ivory", "cream", "off-white"]},
    "gray": {"rgb": (128, 128, 128), "synonyms": ["grey", "charcoal", "slate"]},
    "charcoal": {"rgb": (54, 69, 79), "synonyms": ["dark gray", "anthracite", "graphite"]},
    "silver": {"rgb": (192, 192, 192), "synonyms": ["metallic", "light gray", "pewter"]},
    "cream": {"rgb": (255, 253, 208), "synonyms": ["ivory", "off-white", "vanilla"]},
}


# ============================================================================
# TEXTURE VOCABULARY - Fashion-relevant texture terms
# ============================================================================

TEXTURE_LABELS: Dict[str, List[str]] = {
    "solid": ["plain", "one color", "uniform", "block color"],
    "striped": ["stripe", "pinstripe", "lines", "linear"],
    "plaid": ["checkered", "tartan", "gingham", "checker"],
    "floral": ["flower", "botanical", "garden", "roses"],
    "animal print": ["leopard", "zebra", "snake", "crocodile", "reptile", "exotic"],
    "geometric": ["abstract", "triangles", "squares", "modern"],
    "textured": ["embossed", "raised", "woven", "knit", "ribbed"],
    "polka dot": ["dots", "spotted", "dotted"],
    "paisley": ["swirl", "ornate", "bohemian"],
    "camouflage": ["camo", "military", "hunting"],
    "denim": ["jeans", "chambray", "washed"],
    "leather": ["faux leather", "vegan leather", "genuine leather"],
    "velvet": ["crushed velvet", "plush", "soft"],
    "silk": ["satin", "silky", "shiny", "lustrous"],
    "lace": ["sheer", "delicate", "openwork"],
    "knit": ["cable knit", "sweater", "chunky", "woven"],
    "quilted": ["padded", "puffer", "insulated"],
}


@dataclass
class ColorResult:
    """Result of color extraction."""
    hex: str
    name: str
    synonyms: List[str]
    rgb: Tuple[int, int, int]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OCRResult:
    """Result of OCR text extraction."""
    raw_text: str
    brand: Optional[str]
    product_name: Optional[str]
    price: Optional[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AttributeResult:
    """Complete attribute extraction result."""
    caption: str
    colors: Dict[str, ColorResult]  # primary, secondary
    textures: List[str]
    category: str
    search_queries: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "caption": self.caption,
            "colors": {k: v.to_dict() for k, v in self.colors.items()},
            "textures": self.textures,
            "category": self.category,
            "search_queries": self.search_queries
        }


class AttributeExtractor:
    """
    Multi-modal attribute extractor for fashion images.
    Extracts colors, textures, OCR text, and generates captions.
    """
    
    # Known brand names for OCR matching
    KNOWN_BRANDS = {
        "nike", "adidas", "zara", "h&m", "gucci", "prada", "louis vuitton",
        "chanel", "dior", "versace", "burberry", "ralph lauren", "tommy hilfiger",
        "calvin klein", "levi's", "gap", "uniqlo", "forever 21", "topshop",
        "asos", "mango", "massimo dutti", "cos", "pull&bear", "bershka",
        "stradivarius", "primark", "shein", "nordstrom", "amazon", "walmart",
        "target", "macys", "sephora", "ulta", "lululemon", "athleta",
        "under armour", "puma", "reebok", "new balance", "vans", "converse",
        "jordan", "yeezy", "balenciaga", "fendi", "hermes", "coach", "michael kors"
    }
    
    def __init__(self, use_blip: bool = True, use_ocr: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_blip = use_blip
        self.use_ocr = use_ocr
        
        # Initialize models lazily
        self._blip_processor = None
        self._blip_model = None
        self._ocr_reader = None
        
        logger.info(f"AttributeExtractor initialized on device: {self.device}")
    
    def _load_blip(self):
        """Lazy load BLIP model."""
        if self._blip_model is None and self.use_blip:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                logger.info("Loading BLIP model...")
                self._blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                )
                self._blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                self._blip_model.eval()
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BLIP model: {e}")
                self.use_blip = False
    
    def _load_ocr(self):
        """Lazy load EasyOCR reader."""
        if self._ocr_reader is None and self.use_ocr:
            try:
                import easyocr
                logger.info("Loading EasyOCR reader...")
                # GPU if available, English language
                self._ocr_reader = easyocr.Reader(
                    ['en'], 
                    gpu=torch.cuda.is_available(),
                    verbose=False
                )
                logger.info("EasyOCR reader loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                self.use_ocr = False
    
    def _extract_text_ocr(self, image: Image.Image) -> OCRResult:
        """Extract text from image using OCR."""
        if not self.use_ocr:
            return OCRResult(raw_text="", brand=None, product_name=None, price=None)
        
        self._load_ocr()
        
        if self._ocr_reader is None:
            return OCRResult(raw_text="", brand=None, product_name=None, price=None)
        
        try:
            # Convert PIL to numpy for EasyOCR
            img_array = np.array(image)
            
            # Run OCR
            results = self._ocr_reader.readtext(img_array)
            
            # Combine all detected text
            texts = [result[1] for result in results if result[2] > 0.3]  # Confidence > 30%
            raw_text = " ".join(texts)
            
            logger.info(f"OCR extracted: {raw_text[:100]}...")
            
            # Parse the text
            return self._parse_ocr_text(raw_text)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(raw_text="", brand=None, product_name=None, price=None)
    
    def _parse_ocr_text(self, raw_text: str) -> OCRResult:
        """Parse OCR text to extract brand, product name, and price."""
        import re
        
        brand = None
        product_name = None
        price = None
        
        text_lower = raw_text.lower()
        
        # Find brand
        for known_brand in self.KNOWN_BRANDS:
            if known_brand in text_lower:
                brand = known_brand.title()
                break
        
        # Find price (patterns like $19.99, $199, €50, £30)
        price_patterns = [
            r'[\$€£]\s*\d+\.?\d*',  # $19.99
            r'\d+\.?\d*\s*[\$€£]',  # 19.99$
            r'\d+\.\d{2}',          # 19.99 (assume currency)
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, raw_text)
            if match:
                price = match.group().strip()
                break
        
        # Product name: longest text segment that's not price/brand
        words = raw_text.split()
        product_candidates = []
        for word in words:
            # Skip if it's a price or brand
            if word.lower() in self.KNOWN_BRANDS:
                continue
            if re.match(r'[\$€£]?\d+\.?\d*', word):
                continue
            if len(word) > 2:
                product_candidates.append(word)
        
        if product_candidates:
            product_name = " ".join(product_candidates[:5])  # First 5 words
        
        return OCRResult(
            raw_text=raw_text,
            brand=brand,
            product_name=product_name,
            price=price
        )
    
    def _preprocess_image(self, image: Union[str, bytes, Image.Image]) -> Image.Image:
        """Convert input to PIL Image."""
        if isinstance(image, str):
            # Base64 string
            image_bytes = base64.b64decode(image)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using BLIP."""
        if not self.use_blip:
            return ""
        
        self._load_blip()
        
        if self._blip_model is None:
            return ""
        
        try:
            # Use fashion-focused prompt
            inputs = self._blip_processor(
                image, 
                text="a photo of",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output = self._blip_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
            
            caption = self._blip_processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")
            return ""
    
    def extract_colors(
        self, 
        image: Image.Image, 
        num_colors: int = 3
    ) -> Dict[str, ColorResult]:
        """Extract dominant colors using K-Means clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Resize for faster processing
            img = image.copy()
            img.thumbnail((150, 150))
            
            # Convert to numpy array
            pixels = np.array(img).reshape(-1, 3)
            
            # Remove very dark/light pixels (likely background)
            mask = (pixels.mean(axis=1) > 20) & (pixels.mean(axis=1) < 235)
            pixels = pixels[mask]
            
            if len(pixels) < num_colors:
                pixels = np.array(img).reshape(-1, 3)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers and counts
            colors = kmeans.cluster_centers_.astype(int)
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            
            # Sort by frequency
            sorted_indices = np.argsort(-counts)
            
            result = {}
            for i, idx in enumerate(sorted_indices[:2]):  # Primary + Secondary
                rgb = tuple(colors[idx])
                name, synonyms = self._match_color_name(rgb)
                hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
                
                key = "primary" if i == 0 else "secondary"
                result[key] = ColorResult(
                    hex=hex_color,
                    name=name,
                    synonyms=synonyms,
                    rgb=rgb
                )
            
            return result
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return {}
    
    def _match_color_name(self, rgb: Tuple[int, int, int]) -> Tuple[str, List[str]]:
        """Match RGB to nearest color name."""
        min_distance = float('inf')
        best_match = "unknown"
        best_synonyms = []
        
        for name, data in COLOR_MAP.items():
            ref_rgb = data["rgb"]
            # Euclidean distance in RGB space
            distance = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb)) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match = name
                best_synonyms = data["synonyms"]
        
        return best_match, best_synonyms
    
    def detect_textures(self, image: Image.Image) -> List[str]:
        """Detect textures using LBP (Local Binary Patterns)."""
        try:
            from skimage.feature import local_binary_pattern
            from skimage.color import rgb2gray
            
            # Convert to grayscale
            img_array = np.array(image)
            gray = rgb2gray(img_array)
            
            # Resize for faster processing
            from skimage.transform import resize
            gray = resize(gray, (150, 150), anti_aliasing=True)
            
            # Compute LBP
            radius = 2
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Compute histogram of LBP
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            # Analyze pattern characteristics
            textures = []
            
            # High uniformity = solid color
            if hist[n_points] > 0.5:
                textures.append("solid")
            
            # Check for patterns using variance
            variance = np.var(lbp)
            
            if variance > 50:
                textures.append("textured")
            
            if variance > 100:
                textures.append("patterned")
            
            # Check for edge patterns (stripes)
            edge_std = np.std(gray, axis=1).mean()
            if edge_std > 0.1:
                textures.append("striped")
            
            # If no specific texture detected
            if not textures:
                textures.append("solid")
            
            return textures[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"Texture detection failed: {e}")
            return ["unknown"]
    
    def _detect_category(self, caption: str) -> str:
        """Extract product category from caption."""
        caption_lower = caption.lower()
        
        categories = {
            "shirt": ["shirt", "blouse", "top", "tee", "t-shirt", "polo"],
            "dress": ["dress", "gown", "frock"],
            "pants": ["pants", "jeans", "trousers", "shorts", "leggings"],
            "jacket": ["jacket", "coat", "blazer", "hoodie", "cardigan"],
            "shoes": ["shoes", "sneakers", "boots", "heels", "sandals", "loafers"],
            "bag": ["bag", "purse", "handbag", "backpack", "tote"],
            "accessory": ["hat", "scarf", "belt", "watch", "jewelry", "sunglasses"],
            "skirt": ["skirt", "mini", "maxi"],
            "sweater": ["sweater", "pullover", "jumper", "knit"],
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in caption_lower:
                    return category
        
        return "clothing"
    
    def generate_search_queries(
        self, 
        caption: str, 
        colors: Dict[str, ColorResult], 
        textures: List[str],
        category: str
    ) -> List[str]:
        """Generate multiple search query variations."""
        queries = []
        
        primary_color = colors.get("primary")
        color_name = primary_color.name if primary_color else ""
        color_synonyms = primary_color.synonyms if primary_color else []
        
        texture = textures[0] if textures else ""
        
        # Primary query: color + texture + category
        if color_name and texture and category:
            queries.append(f"{color_name} {texture} {category}")
        
        # Category + color
        if color_name and category:
            queries.append(f"{color_name} {category}")
        
        # Using color synonyms
        for syn in color_synonyms[:2]:
            queries.append(f"{syn} {category}")
        
        # From caption (extract key terms)
        if caption:
            # Use caption as-is but cleaned
            clean_caption = caption.replace("a photo of", "").strip()
            if clean_caption and clean_caption not in queries:
                queries.append(clean_caption)
        
        # Texture-focused query
        if texture != "solid" and texture != "unknown":
            queries.append(f"{texture} {category}")
        
        return list(dict.fromkeys(queries))[:5]  # Unique, max 5
    
    def extract(self, image: Union[str, bytes, Image.Image]) -> AttributeResult:
        """
        Extract all attributes from an image using hybrid OCR + Visual AI.
        
        Args:
            image: Base64 string, bytes, or PIL Image
            
        Returns:
            AttributeResult with caption, colors, textures, OCR, and search queries
        """
        # Preprocess
        pil_image = self._preprocess_image(image)
        
        # Extract visual attributes
        caption = self.generate_caption(pil_image)
        colors = self.extract_colors(pil_image)
        textures = self.detect_textures(pil_image)
        category = self._detect_category(caption)
        
        # Extract text via OCR
        ocr_result = self._extract_text_ocr(pil_image)
        
        # Generate search queries (combining visual + OCR)
        search_queries = self._generate_hybrid_queries(
            caption, colors, textures, category, ocr_result
        )
        
        return AttributeResult(
            caption=caption,
            colors=colors,
            textures=textures,
            category=category,
            search_queries=search_queries,
            ocr=ocr_result
        )
    
    def _generate_hybrid_queries(
        self, 
        caption: str, 
        colors: Dict[str, ColorResult], 
        textures: List[str],
        category: str,
        ocr: OCRResult
    ) -> List[str]:
        """Generate search queries combining visual AI and OCR results."""
        queries = []
        
        primary_color = colors.get("primary")
        color_name = primary_color.name if primary_color else ""
        texture = textures[0] if textures else ""
        
        # Priority 1: OCR brand + visual description (most accurate)
        if ocr.brand:
            if color_name and category:
                queries.append(f"{ocr.brand} {color_name} {category}")
            if category:
                queries.append(f"{ocr.brand} {category}")
            # Brand + OCR product name
            if ocr.product_name:
                queries.append(f"{ocr.brand} {ocr.product_name}")
        
        # Priority 2: Visual AI queries
        if color_name and texture and category:
            queries.append(f"{color_name} {texture} {category}")
        
        if color_name and category:
            queries.append(f"{color_name} {category}")
        
        # Priority 3: Caption-based
        if caption:
            clean_caption = caption.replace("a photo of", "").strip()
            if clean_caption and clean_caption not in queries:
                queries.append(clean_caption)
        
        # Priority 4: OCR product name alone
        if ocr.product_name and ocr.product_name not in queries:
            queries.append(ocr.product_name)
        
        return list(dict.fromkeys(queries))[:6]  # Unique, max 6


# Singleton instance
_extractor_instance = None


def get_attribute_extractor(use_blip: bool = True) -> AttributeExtractor:
    """Get singleton attribute extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = AttributeExtractor(use_blip=use_blip)
    return _extractor_instance
