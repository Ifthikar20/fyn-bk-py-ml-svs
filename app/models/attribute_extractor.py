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
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
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
    
    def __init__(self, use_blip: bool = False, use_ocr: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_blip = use_blip  # Disabled by default — too heavy for CPU
        self.use_ocr = use_ocr
        
        # Initialize models lazily
        self._blip_processor = None
        self._blip_model = None
        self._ocr_reader = None
        self._efficientnet = None
        self._efficientnet_transforms = None
        
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
    
    def _preprocess_image(self, image: Union[str, bytes, Image.Image], max_size: int = 800) -> Image.Image:
        """Convert input to PIL Image and resize for speed."""
        if isinstance(image, str):
            # Base64 string
            image_bytes = base64.b64decode(image)
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize large images for faster processing
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.LANCZOS)
            logger.info(f"Resized image to {image.size}")
        
        return image
    
    def _load_efficientnet(self):
        """Lazy load EfficientNet for fast image classification."""
        if self._efficientnet is None:
            try:
                from torchvision import models, transforms
                logger.info("Loading EfficientNet-B0 for classification...")
                self._efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                self._efficientnet.eval()
                self._efficientnet.to(self.device)
                self._efficientnet_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                logger.info("EfficientNet-B0 loaded for classification")
            except Exception as e:
                logger.error(f"Failed to load EfficientNet: {e}")
    
    # Fashion-relevant ImageNet class mappings
    IMAGENET_FASHION_MAP = {
        # Clothing
        "jersey": "jersey top", "suit": "suit", "academic_gown": "gown",
        "jean": "jeans", "swimming_trunks": "swim trunks", "bikini": "bikini",
        "miniskirt": "mini skirt", "overskirt": "skirt", "hoopskirt": "hoop skirt",
        "sarong": "sarong", "kimono": "kimono", "abaya": "abaya",
        "poncho": "poncho", "cardigan": "cardigan", "sweatshirt": "sweatshirt",
        "trench_coat": "trench coat", "lab_coat": "coat", "fur_coat": "fur coat",
        "bulletproof_vest": "vest", "brassiere": "bra", "pajama": "pajamas",
        "bow_tie": "bow tie", "neck_brace": "necklace",
        # Shoes
        "running_shoe": "running shoes", "clog": "clogs", "sandal": "sandals",
        "cowboy_boot": "cowboy boots", "Loafer": "loafers",
        # Bags
        "backpack": "backpack", "purse": "purse", "handbag": "handbag",
        "shopping_basket": "tote bag", "mailbag": "messenger bag",
        "plastic_bag": "bag",
        # Accessories
        "sunglass": "sunglasses", "sunglasses": "sunglasses",
        "watch": "watch", "digital_watch": "digital watch",
        "necklace": "necklace", "hat": "hat", "sombrero": "hat",
        "cowboy_hat": "cowboy hat", "bonnet": "bonnet",
        "shower_cap": "cap", "swimming_cap": "cap",
        "ski_mask": "mask", "bolo_tie": "bolo tie",
        "scarf": "scarf", "stole": "stole",
        "wallet": "wallet", "pencil_case": "pouch",
        # Person-related labels (EfficientNet often returns these for people wearing clothes)
        "nematode": "clothing", "lipstick": "clothing", "face_powder": "clothing",
        "makeup": "clothing", "wig": "clothing", "mask": "clothing",
        "hair_slide": "clothing", "Band_Aid": "clothing",
        "shower_curtain": "clothing", "window_shade": "clothing",
        "spotlight": "clothing", "desk": "clothing", "screen": "clothing",
        # Other products
        "perfume": "perfume", "lotion": "lotion",
        "hair_spray": "hair product",
        "pillow": "pillow", "quilt": "quilt", "blanket": "blanket",
        "teddy": "stuffed toy", "candle": "candle",
        "water_bottle": "water bottle", "beer_bottle": "bottle",
        "wine_bottle": "wine", "coffee_mug": "mug", "cup": "cup",
        "plate": "plate", "bowl": "bowl",
    }
    
    # Map nuanced colors to simpler, more searchable terms
    COLOR_SIMPLIFY_MAP = {
        "teal": "green", "cyan": "blue", "aqua": "blue", "turquoise": "green",
        "magenta": "pink", "fuchsia": "pink", "crimson": "red", "scarlet": "red",
        "burgundy": "red", "maroon": "red", "coral": "pink", "salmon": "pink",
        "navy": "blue", "indigo": "blue", "cobalt": "blue", "azure": "blue",
        "olive": "green", "lime": "green", "sage": "green", "mint": "green",
        "khaki": "beige", "tan": "brown", "camel": "brown", "sand": "beige",
        "charcoal": "gray", "slate": "gray", "silver": "gray",
        "ivory": "white", "cream": "white", "eggshell": "white",
        "plum": "purple", "lavender": "purple", "violet": "purple",
        "mauve": "purple", "lilac": "purple",
        "gold": "yellow", "amber": "yellow", "mustard": "yellow",
        "rust": "orange", "copper": "orange", "peach": "orange",
    }
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using EfficientNet classification (fast, <1s on CPU)."""
        # Use BLIP if available (GPU environments)
        if self.use_blip:
            self._load_blip()
            if self._blip_model is not None:
                try:
                    blip_image = image.copy()
                    blip_image.thumbnail((384, 384), Image.LANCZOS)
                    inputs = self._blip_processor(
                        blip_image, text="a photo of", return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        output = self._blip_model.generate(
                            **inputs, max_length=30, num_beams=1, do_sample=False,
                        )
                    return self._blip_processor.decode(output[0], skip_special_tokens=True)
                except Exception as e:
                    logger.error(f"BLIP caption failed: {e}")
        
        # Fast path: EfficientNet ImageNet classification (<1s on CPU)
        self._load_efficientnet()
        if self._efficientnet is None:
            return ""
        
        try:
            img_tensor = self._efficientnet_transforms(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self._efficientnet(img_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_idx = torch.topk(probs, 5)
            
            # Get ImageNet class names
            from torchvision.models import EfficientNet_B0_Weights
            categories = EfficientNet_B0_Weights.IMAGENET1K_V1.meta["categories"]
            
            # Find best fashion-relevant label
            for prob, idx in zip(top5_prob, top5_idx):
                label = categories[idx.item()]
                label_clean = label.replace(" ", "_").lower()
                
                # Check fashion map first
                for key, fashion_name in self.IMAGENET_FASHION_MAP.items():
                    if key.lower() in label_clean or label_clean in key.lower():
                        confidence = prob.item()
                        logger.info(f"EfficientNet: {label} ({confidence:.1%}) -> {fashion_name}")
                        return f"a photo of {fashion_name}"
            
            # Fallback: use top-1 label as-is
            top_label = categories[top5_idx[0].item()].replace("_", " ")
            top_conf = top5_prob[0].item()
            logger.info(f"EfficientNet top-1: {top_label} ({top_conf:.1%})")
            return f"a photo of {top_label}"
            
        except Exception as e:
            logger.error(f"EfficientNet caption failed: {e}")
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
        """Generate smart search query variations optimized for fashion search."""
        queries = []
        
        primary_color = colors.get("primary")
        color_name = primary_color.name if primary_color else ""
        color_synonyms = primary_color.synonyms if primary_color else []
        
        # Simplify color for better search results (teal → green, etc.)
        simple_color = self.COLOR_SIMPLIFY_MAP.get(color_name.lower(), color_name) if color_name else ""
        
        texture = textures[0] if textures else ""
        skip_texture = texture in ("solid", "unknown", "")
        
        # When category is generic "clothing", generate garment-specific queries
        # This handles cases where EfficientNet can't identify the specific garment type
        if category == "clothing" and simple_color:
            # Most common garment types to try
            garment_types = ["shirt", "top", "dress"]
            for garment in garment_types:
                queries.append(f"{simple_color} {garment}")
            # Also try with original nuanced color for variety
            if color_name.lower() != simple_color.lower():
                queries.append(f"{color_name} shirt")
        elif category != "clothing":
            # Specific category: use it directly
            if simple_color and not skip_texture:
                queries.append(f"{simple_color} {texture} {category}")
            if simple_color:
                queries.append(f"{simple_color} {category}")
            queries.append(category)
        
        # From caption (only if it's a specific item, not generic classifications)
        if caption:
            clean_caption = caption.replace("a photo of", "").strip()
            generic_terms = {"clothing", "makeup", "nematode", "mask", "wig"}
            if clean_caption and clean_caption.lower() not in generic_terms:
                if clean_caption not in queries:
                    queries.append(clean_caption)
        
        # Texture-focused query (only for specific, non-generic categories)
        if not skip_texture and category != "clothing":
            queries.append(f"{texture} {category}")
        
        # Ensure at least one query exists
        if not queries:
            if simple_color:
                queries.append(f"{simple_color} fashion")
            elif caption:
                queries.append(caption.replace("a photo of", "").strip())
            else:
                queries.append("fashion")
        
        return list(dict.fromkeys(queries))[:5]  # Unique, max 5
    
    def extract(self, image: Union[str, bytes, Image.Image], skip_ocr: bool = True) -> AttributeResult:
        """
        Extract all attributes from an image using hybrid OCR + Visual AI.
        
        Args:
            image: Base64 string, bytes, or PIL Image
            skip_ocr: Skip OCR for speed (default True — saves ~8s on CPU)
            
        Returns:
            AttributeResult with caption, colors, textures, and search queries
        """
        t0 = time.time()
        
        # Preprocess (includes resize to max 800px)
        pil_image = self._preprocess_image(image)
        logger.info(f"Preprocess: {(time.time()-t0)*1000:.0f}ms")
        
        # Run caption, colors, textures IN PARALLEL (they're independent)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            caption_future = executor.submit(self.generate_caption, pil_image)
            colors_future = executor.submit(self.extract_colors, pil_image)
            textures_future = executor.submit(self.detect_textures, pil_image)
            
            caption = caption_future.result()
            colors = colors_future.result()
            textures = textures_future.result()
        
        logger.info(f"Parallel extraction: {(time.time()-t1)*1000:.0f}ms")
        
        category = self._detect_category(caption)
        
        # OCR is optional — skip by default for speed (~8s savings on CPU)
        if skip_ocr:
            ocr_result = OCRResult(raw_text="", brand=None, product_name=None, price=None)
        else:
            ocr_result = self._extract_text_ocr(pil_image)
        
        # Generate search queries (combining visual + OCR if available)
        search_queries = self._generate_hybrid_queries(
            caption, colors, textures, category, ocr_result
        )
        
        logger.info(f"Total extract: {(time.time()-t0)*1000:.0f}ms")
        
        return AttributeResult(
            caption=caption,
            colors=colors,
            textures=textures,
            category=category,
            search_queries=search_queries,
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
        skip_texture = texture in ("solid", "unknown", "")
        
        # Simplify color for better search results (teal → green, etc.)
        simple_color = self.COLOR_SIMPLIFY_MAP.get(color_name.lower(), color_name) if color_name else ""
        
        # Priority 1: OCR brand + visual description (most accurate)
        if ocr.brand:
            if simple_color and category:
                queries.append(f"{ocr.brand} {simple_color} {category}")
            if category:
                queries.append(f"{ocr.brand} {category}")
            if ocr.product_name:
                queries.append(f"{ocr.brand} {ocr.product_name}")
        
        # Priority 2: Visual AI queries — handle generic "clothing" specially
        if category == "clothing" and simple_color:
            # EfficientNet can't distinguish garment types — try common ones
            for garment in ["shirt", "top", "dress"]:
                queries.append(f"{simple_color} {garment}")
            # Also try with original nuanced color
            if color_name.lower() != simple_color.lower():
                queries.append(f"{color_name} shirt")
        elif category != "clothing":
            # Specific category: use it directly
            if simple_color and not skip_texture:
                queries.append(f"{simple_color} {texture} {category}")
            if simple_color:
                queries.append(f"{simple_color} {category}")
            queries.append(category)
        
        # Priority 3: Caption-based (skip generic/irrelevant captions)
        if caption:
            clean_caption = caption.replace("a photo of", "").strip()
            generic_terms = {"clothing", "makeup", "nematode", "mask", "wig", 
                           "face_powder", "lipstick", "screen", "desk"}
            if clean_caption and clean_caption.lower() not in generic_terms:
                if clean_caption not in queries:
                    queries.append(clean_caption)
        
        # Priority 4: OCR product name alone
        if ocr.product_name and ocr.product_name not in queries:
            queries.append(ocr.product_name)
        
        # Ensure at least one query
        if not queries:
            if simple_color:
                queries.append(f"{simple_color} fashion")
            elif caption:
                queries.append(caption.replace("a photo of", "").strip())
            else:
                queries.append("fashion")
        
        return list(dict.fromkeys(queries))[:6]  # Unique, max 6


# Singleton instance
_extractor_instance = None


def get_attribute_extractor(use_blip: bool = False) -> AttributeExtractor:
    """Get singleton attribute extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = AttributeExtractor(use_blip=use_blip)
    return _extractor_instance
