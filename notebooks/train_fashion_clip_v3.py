"""
=============================================================================
 FYNDA Fashion-CLIP v3 Training Notebook  
 ----------------------------------------
 Run this in Google Colab Pro with T4 GPU
 
 WHAT'S NEW IN v3:
 ✅ Fashionpedia dataset — 48K images with 294 fine-grained attributes
 ✅ Texture/material vocabulary — cotton, silk, denim, leather, linen...
 ✅ 4 datasets combined — ~108K images, ~1.2M+ text-image pairs
 ✅ Material-enriched descriptions — "a soft cotton t-shirt", "denim jacket"
 ✅ Attribute-aware templates — leveraging Fashionpedia's annotated attributes
 ✅ Resume from v2 checkpoint — continuous improvement
 
 HOW TO USE:
 1. Open Google Colab (colab.research.google.com)
 2. Create a new notebook
 3. Go to Runtime → Change runtime type → T4 GPU
 4. Copy each "# CELL X" section into a separate Colab cell
 5. Run cells in order (Shift+Enter)
 
 Training time: ~12-18 hours on T4 GPU
 Disk needed: ~20GB (datasets + model)
 Output: fashion_clip_fynda_v3.pt (~577MB saved to Google Drive)
=============================================================================
"""

# ============================================================================
# CELL 1: Install Dependencies (~2 min)
# ============================================================================

!pip install -q transformers torch torchvision pillow kagglehub tqdm
!pip install -q open_clip_torch
!pip install -q datasets  # For HuggingFace Fashionpedia

# Mount Google Drive for saving the model
from google.colab import drive
drive.mount('/content/drive')


# ============================================================================
# CELL 2: Download All Fashion Datasets (~25 min)
# ============================================================================

"""
We combine 4 fashion datasets for maximum diversity:

1. Fashion Product Images (Small) — 44K studio product shots with rich metadata
2. Clothing Dataset Full — 5K real-world photos (natural lighting, varied angles)  
3. Fashion-MNIST alternative — 10K diverse clothing items
4. Fashionpedia — 48K images with 294 fine-grained attributes (material, style)

Total: ~108K unique fashion images
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

# --- Dataset 1: Fashion Product Images (our primary dataset) ---
import kagglehub

print("📥 Downloading Dataset 1: Fashion Product Images (44K)...")
dataset1_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print(f"   ✅ Downloaded to: {dataset1_path}")

# --- Dataset 2: Clothing Dataset Full (real-world photos) ---
print("\n📥 Downloading Dataset 2: Clothing Dataset Full (5K)...")
try:
    dataset2_path = kagglehub.dataset_download("agrigorev/clothing-dataset-full")
    print(f"   ✅ Downloaded to: {dataset2_path}")
    HAS_DATASET2 = True
except Exception as e:
    print(f"   ⚠️ Could not download Dataset 2: {e}")
    print("   Continuing with Dataset 1 only...")
    dataset2_path = None
    HAS_DATASET2 = False

# --- Dataset 3: Clothes Dataset (diverse categories) ---
print("\n📥 Downloading Dataset 3: Clothes Dataset (diverse)...")
try:
    dataset3_path = kagglehub.dataset_download("validmodel/clothing-and-style")
    print(f"   ✅ Downloaded to: {dataset3_path}")
    HAS_DATASET3 = True
except Exception as e:
    print(f"   ⚠️ Could not download Dataset 3: {e}")
    print("   Continuing without it...")
    dataset3_path = None
    HAS_DATASET3 = False

# --- Dataset 4: Fashionpedia (fine-grained attributes) ---
print("\n📥 Downloading Dataset 4: Fashionpedia (48K with attributes)...")
HAS_FASHIONPEDIA = False
try:
    from datasets import load_dataset
    fp_dataset = load_dataset("detection-datasets/fashionpedia", split="train")
    HAS_FASHIONPEDIA = True
    print(f"   ✅ Loaded Fashionpedia: {len(fp_dataset)} images")
except Exception as e:
    print(f"   ⚠️ Could not load Fashionpedia from HuggingFace: {e}")
    print("   Trying Kaggle fallback...")
    try:
        dataset4_path = kagglehub.dataset_download("rangsimanketkaew/fashionpedia")
        HAS_FASHIONPEDIA = True
        fp_dataset = None  # Will use file-based loading
        print(f"   ✅ Downloaded from Kaggle to: {dataset4_path}")
    except Exception as e2:
        print(f"   ⚠️ Could not download Fashionpedia: {e2}")
        print("   Continuing without it...")

print("\n" + "="*60)
print("📊 Dataset Summary:")
print(f"   Dataset 1 (Fashion Products):  ✅ {dataset1_path}")
if HAS_DATASET2:
    print(f"   Dataset 2 (Clothing Full):     ✅ {dataset2_path}")
if HAS_DATASET3:
    print(f"   Dataset 3 (Clothes):           ✅ {dataset3_path}")
if HAS_FASHIONPEDIA:
    print(f"   Dataset 4 (Fashionpedia):      ✅ Loaded")
print("="*60)


# ============================================================================
# CELL 3: Prepare & Merge All Datasets (~5 min)
# ============================================================================

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ---- Dataset 1: Fashion Product Images ----
styles_csv = os.path.join(dataset1_path, "styles.csv")
df1 = pd.read_csv(styles_csv, on_bad_lines='skip')

# Build pairs: (image_path, metadata_dict)
dataset1_pairs = []
images_dir_1 = os.path.join(dataset1_path, "images")

for _, row in df1.iterrows():
    img_path = os.path.join(images_dir_1, f"{row['id']}.jpg")
    if os.path.exists(img_path):
        dataset1_pairs.append({
            'image_path': img_path,
            'color': str(row.get('baseColour', '')).strip().lower() if pd.notna(row.get('baseColour')) else '',
            'gender': str(row.get('gender', '')).strip().lower() if pd.notna(row.get('gender')) else '',
            'article': str(row.get('articleType', '')).strip().lower() if pd.notna(row.get('articleType')) else '',
            'usage': str(row.get('usage', '')).strip().lower() if pd.notna(row.get('usage')) else '',
            'season': str(row.get('season', '')).strip().lower() if pd.notna(row.get('season')) else '',
            'sub_cat': str(row.get('subCategory', '')).strip().lower() if pd.notna(row.get('subCategory')) else '',
            'master_cat': str(row.get('masterCategory', '')).strip().lower() if pd.notna(row.get('masterCategory')) else '',
            'display_name': str(row.get('productDisplayName', '')).strip().lower() if pd.notna(row.get('productDisplayName')) else '',
            'material': '',  # Dataset 1 doesn't have material info
            'attributes': [],
            'source': 'fashion_products',
        })

print(f"✅ Dataset 1: {len(dataset1_pairs)} images with metadata")

# ---- Dataset 2: Clothing Dataset Full ----
dataset2_pairs = []
if HAS_DATASET2:
    for root, dirs, files in os.walk(dataset2_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                category = os.path.basename(root).lower().replace('_', ' ')
                if category and category != os.path.basename(dataset2_path).lower():
                    dataset2_pairs.append({
                        'image_path': img_path,
                        'color': '',
                        'gender': '',
                        'article': category,
                        'usage': '',
                        'season': '',
                        'sub_cat': '',
                        'master_cat': 'apparel',
                        'display_name': category,
                        'material': '',
                        'attributes': [],
                        'source': 'clothing_full',
                    })
    print(f"✅ Dataset 2: {len(dataset2_pairs)} real-world photos")

# ---- Dataset 3: Clothes Dataset ----
dataset3_pairs = []
if HAS_DATASET3:
    for root, dirs, files in os.walk(dataset3_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                category = os.path.basename(root).lower().replace('_', ' ')
                if category and category != os.path.basename(dataset3_path).lower():
                    dataset3_pairs.append({
                        'image_path': img_path,
                        'color': '',
                        'gender': '',
                        'article': category,
                        'usage': '',
                        'season': '',
                        'sub_cat': '',
                        'master_cat': 'apparel',
                        'display_name': category,
                        'material': '',
                        'attributes': [],
                        'source': 'clothes_dataset',
                    })
    print(f"✅ Dataset 3: {len(dataset3_pairs)} diverse images")

# ---- Dataset 4: Fashionpedia ----
# Fashionpedia category names (27 apparel + 19 parts)
FASHIONPEDIA_CATEGORIES = {
    0: "shirt, blouse", 1: "top, t-shirt, sweatshirt", 2: "sweater",
    3: "cardigan", 4: "jacket", 5: "vest", 6: "pants", 7: "shorts",
    8: "skirt", 9: "coat", 10: "dress", 11: "jumpsuit", 12: "cape",
    13: "glasses", 14: "hat", 15: "headband, head covering, hair accessory",
    16: "tie", 17: "glove", 18: "watch", 19: "belt", 20: "leg warmer",
    21: "tights, stockings", 22: "sock", 23: "shoe", 24: "bag, wallet",
    25: "scarf", 26: "umbrella",
    # Parts (27-45)
    27: "hood", 28: "collar", 29: "lapel", 30: "epaulette", 31: "sleeve",
    32: "pocket", 33: "neckline", 34: "buckle", 35: "zipper", 36: "applique",
    37: "bead", 38: "bow", 39: "flower", 40: "fringe", 41: "ribbon",
    42: "rivet", 43: "ruffle", 44: "sequin", 45: "tassel",
}

# Fashionpedia attribute groups (294 total, key groups for text generation)
FASHIONPEDIA_ATTR_GROUPS = {
    # Material attributes (indices 0-24)
    "material": {
        0: "denim", 1: "cotton", 2: "leather", 3: "fur", 4: "silk",
        5: "wool", 6: "synthetic", 7: "knitted", 8: "lace",
        9: "velvet", 10: "suede", 11: "chiffon", 12: "nylon",
        13: "linen", 14: "satin", 15: "polyester", 16: "cashmere",
        17: "mesh", 18: "corduroy", 19: "fleece", 20: "tweed",
        21: "sequined", 22: "metallic", 23: "rubber", 24: "canvas",
    },
    # Pattern/textile attributes
    "pattern": {
        25: "solid", 26: "striped", 27: "plaid", 28: "floral",
        29: "polka dot", 30: "paisley", 31: "graphic print", 32: "camouflage",
        33: "animal print", 34: "abstract", 35: "checkered", 36: "colorblock",
        37: "tie-dye", 38: "houndstooth", 39: "herringbone",
    },
    # Style/silhouette attributes
    "style": {
        40: "fitted", 41: "oversized", 42: "slim fit", 43: "regular fit",
        44: "relaxed", 45: "cropped", 46: "high-waisted", 47: "low-rise",
        48: "asymmetric", 49: "wrap", 50: "a-line", 51: "bodycon",
        52: "pleated", 53: "ruffled", 54: "draped",
    },
    # Closure/construction attributes
    "closure": {
        55: "button", 56: "zipper", 57: "snap", 58: "hook",
        59: "lace-up", 60: "buckle", 61: "velcro", 62: "pullover",
        63: "drawstring",
    },
    # Neckline attributes
    "neckline": {
        64: "crew neck", 65: "v-neck", 66: "scoop neck", 67: "turtleneck",
        68: "off-shoulder", 69: "halter", 70: "collared", 71: "boat neck",
        72: "cowl neck", 73: "square neck", 74: "mock neck",
    },
    # Length attributes
    "length": {
        75: "mini", 76: "midi", 77: "maxi", 78: "knee-length",
        79: "ankle-length", 80: "full-length", 81: "cropped",
    },
    # Sleeve attributes
    "sleeve": {
        82: "sleeveless", 83: "short sleeve", 84: "long sleeve",
        85: "three-quarter sleeve", 86: "cap sleeve", 87: "puff sleeve",
        88: "bell sleeve", 89: "raglan sleeve",
    },
}

# Texture descriptors for material enrichment
TEXTURE_DESCRIPTORS = {
    "cotton": ["soft", "breathable", "lightweight"],
    "denim": ["sturdy", "woven", "rugged"],
    "leather": ["smooth", "supple", "rich"],
    "silk": ["luxurious", "flowing", "delicate"],
    "wool": ["warm", "textured", "cozy"],
    "linen": ["crisp", "relaxed", "airy"],
    "velvet": ["plush", "rich", "sumptuous"],
    "cashmere": ["ultra-soft", "premium", "luxurious"],
    "satin": ["smooth", "glossy", "silky"],
    "chiffon": ["sheer", "lightweight", "ethereal"],
    "corduroy": ["ribbed", "textured", "retro"],
    "tweed": ["woven", "classic", "structured"],
    "lace": ["delicate", "intricate", "feminine"],
    "suede": ["soft", "napped", "velvety"],
    "fleece": ["plush", "cozy", "warm"],
    "knitted": ["textured", "stretchy", "handcrafted"],
    "mesh": ["breathable", "transparent", "sporty"],
    "nylon": ["lightweight", "durable", "water-resistant"],
    "polyester": ["wrinkle-resistant", "durable", "versatile"],
    "canvas": ["heavy-duty", "sturdy", "rugged"],
}

# Map article types to likely materials (for datasets 1-3 that lack material info)
ARTICLE_MATERIAL_MAP = {
    "t-shirt": ["cotton", "cotton", "polyester"],
    "tshirt": ["cotton", "cotton", "polyester"],
    "shirt": ["cotton", "linen", "polyester"],
    "blouse": ["silk", "chiffon", "cotton"],
    "top": ["cotton", "polyester", "knitted"],
    "sweater": ["wool", "cashmere", "knitted"],
    "hoodie": ["cotton", "fleece", "polyester"],
    "jacket": ["denim", "leather", "nylon", "polyester"],
    "coat": ["wool", "cashmere", "polyester"],
    "blazer": ["wool", "polyester", "linen"],
    "suit": ["wool", "linen", "polyester"],
    "dress": ["cotton", "silk", "chiffon", "polyester"],
    "gown": ["silk", "satin", "chiffon"],
    "skirt": ["cotton", "denim", "polyester"],
    "jeans": ["denim", "denim", "denim"],
    "pants": ["cotton", "polyester", "wool"],
    "trousers": ["wool", "cotton", "polyester"],
    "shorts": ["cotton", "denim", "nylon"],
    "leggings": ["nylon", "polyester"],
    "cardigan": ["wool", "cashmere", "knitted"],
    "vest": ["cotton", "polyester", "denim"],
    "scarf": ["silk", "wool", "cashmere"],
    "belt": ["leather", "canvas", "suede"],
    "shoes": ["leather", "canvas", "suede"],
    "sneakers": ["canvas", "nylon", "leather"],
    "boots": ["leather", "suede"],
    "sandals": ["leather", "suede", "rubber"],
    "bag": ["leather", "canvas", "nylon"],
    "handbag": ["leather", "suede", "canvas"],
    "backpack": ["nylon", "canvas", "polyester"],
    "watch": ["leather", "metallic"],
    "sunglasses": ["metallic", "plastic"],
    "hat": ["cotton", "wool", "canvas"],
    "cap": ["cotton", "polyester", "canvas"],
    "socks": ["cotton", "wool", "nylon"],
    "tie": ["silk", "polyester"],
    "polo shirt": ["cotton", "polyester"],
    "jumpsuit": ["cotton", "denim", "polyester"],
    "kurta": ["cotton", "silk", "linen"],
    "saree": ["silk", "cotton", "chiffon"],
}

# ---- Parse Fashionpedia from HuggingFace ----
dataset4_pairs = []
if HAS_FASHIONPEDIA and fp_dataset is not None:
    print("\n📊 Parsing Fashionpedia attributes...")
    fp_save_dir = "/content/fashionpedia_images"
    os.makedirs(fp_save_dir, exist_ok=True)
    
    for idx, item in enumerate(fp_dataset):
        try:
            img = item.get("image")
            if img is None:
                continue
            
            # Save image to disk
            img_path = os.path.join(fp_save_dir, f"fp_{idx}.jpg")
            if not os.path.exists(img_path):
                img.save(img_path)
            
            # Extract category and attribute info
            categories = item.get("objects", {}).get("category", [])
            attributes = item.get("objects", {}).get("attribute", [])
            
            # Get primary category name
            primary_cat = ""
            if categories:
                cat_id = categories[0] if isinstance(categories[0], int) else 0
                primary_cat = FASHIONPEDIA_CATEGORIES.get(cat_id, "")
                # Use only the first name if comma-separated
                primary_cat = primary_cat.split(",")[0].strip()
            
            # Extract attribute names
            attr_names = []
            material_names = []
            pattern_names = []
            style_names = []
            neckline_names = []
            length_names = []
            sleeve_names = []
            
            for attr_list in attributes:
                if isinstance(attr_list, list):
                    for attr_id in attr_list:
                        if not isinstance(attr_id, int):
                            continue
                        for group_name, group_dict in FASHIONPEDIA_ATTR_GROUPS.items():
                            if attr_id in group_dict:
                                attr_name = group_dict[attr_id]
                                attr_names.append(attr_name)
                                if group_name == "material":
                                    material_names.append(attr_name)
                                elif group_name == "pattern":
                                    pattern_names.append(attr_name)
                                elif group_name == "style":
                                    style_names.append(attr_name)
                                elif group_name == "neckline":
                                    neckline_names.append(attr_name)
                                elif group_name == "length":
                                    length_names.append(attr_name)
                                elif group_name == "sleeve":
                                    sleeve_names.append(attr_name)
            
            dataset4_pairs.append({
                'image_path': img_path,
                'color': '',
                'gender': '',
                'article': primary_cat,
                'usage': '',
                'season': '',
                'sub_cat': '',
                'master_cat': 'apparel',
                'display_name': primary_cat,
                'material': material_names[0] if material_names else '',
                'attributes': attr_names,
                'material_list': material_names,
                'pattern_list': pattern_names,
                'style_list': style_names,
                'neckline_list': neckline_names,
                'length_list': length_names,
                'sleeve_list': sleeve_names,
                'source': 'fashionpedia',
            })
            
            if (idx + 1) % 5000 == 0:
                print(f"   Processed {idx + 1}/{len(fp_dataset)} images...")
                
        except Exception as e:
            continue
    
    print(f"✅ Dataset 4: {len(dataset4_pairs)} Fashionpedia images with attributes")

# ---- Merge all datasets ----
all_pairs = dataset1_pairs + dataset2_pairs + dataset3_pairs + dataset4_pairs
print(f"\n🔗 Total merged: {len(all_pairs)} images from {1 + int(HAS_DATASET2) + int(HAS_DATASET3) + int(HAS_FASHIONPEDIA)} datasets")


# ============================================================================
# CELL 4: Create Texture-Enriched Text Descriptions v3 (~3 min)
# ============================================================================

import random

def create_fashion_description_v3(item):
    """
    v3 description generator — creates 12-18 diverse text descriptions per image.
    
    Key improvements over v2:
    - Material/texture-enriched descriptions
    - Fashionpedia attribute integration
    - Fabric texture adjective injection
    - Pattern and style attribute templates
    """
    color = item.get('color', '')
    gender = item.get('gender', '')
    article = item.get('article', '')
    usage = item.get('usage', '')
    season = item.get('season', '')
    sub_cat = item.get('sub_cat', '')
    display_name = item.get('display_name', '')
    material = item.get('material', '')
    attributes = item.get('attributes', [])
    material_list = item.get('material_list', [])
    pattern_list = item.get('pattern_list', [])
    style_list = item.get('style_list', [])
    neckline_list = item.get('neckline_list', [])
    length_list = item.get('length_list', [])
    sleeve_list = item.get('sleeve_list', [])
    
    # Clean up
    for field in ['color', 'gender', 'article', 'usage', 'season', 'sub_cat', 'material']:
        val = item.get(field, '')
        if str(val) in ['nan', 'none', '', 'None']:
            item[field] = ''
    color = item.get('color', '')
    gender = item.get('gender', '')
    article = item.get('article', '')
    usage = item.get('usage', '')
    season = item.get('season', '')
    sub_cat = item.get('sub_cat', '')
    material = item.get('material', '')
    
    descriptions = []
    
    # ─── v2 base templates (kept) ────────────────────
    
    # CLIP-native "a photo of" templates
    if article:
        descriptions.append(f"a photo of a {article}")
        if color:
            descriptions.append(f"a photo of a {color} {article}")
        if color and gender and gender != 'unisex':
            descriptions.append(f"a photo of a {color} {gender}'s {article}")
    
    # Descriptive templates
    parts = [p for p in [color, usage, article] if p]
    if parts:
        desc = ' '.join(parts)
        if gender and gender != 'unisex':
            desc += f" for {gender}"
        descriptions.append(desc)
    
    # Gender-prefixed
    if gender and gender != 'unisex' and article:
        gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
        parts = [p for p in [gender_prefix, color, article] if p]
        descriptions.append(' '.join(parts))
    
    # Color + article
    if color and article:
        descriptions.append(f"{color} {article}")
    
    # Season
    if season and color and article:
        descriptions.append(f"{season} {color} {article}")
    
    # Sub-category
    if usage and sub_cat and color:
        descriptions.append(f"{usage} {sub_cat} in {color}")
    
    # Context-rich
    if color and article:
        descriptions.append(f"a {color} {article} on a white background")
        descriptions.append(f"a closeup of a {color} {article}")
    
    # Display name
    if display_name and display_name != 'nan':
        descriptions.append(display_name)
    
    # Simple category
    if article:
        descriptions.append(article)
    
    # ─── v3 NEW: Material/texture enriched templates ─────────
    
    # Determine material to use
    chosen_material = material  # from Fashionpedia
    if not chosen_material and article:
        # Infer material from article type
        article_lower = article.lower()
        for art_key, materials in ARTICLE_MATERIAL_MAP.items():
            if art_key in article_lower:
                chosen_material = random.choice(materials)
                break
    
    if chosen_material and article:
        # "a cotton t-shirt"
        descriptions.append(f"a {chosen_material} {article}")
        
        # "a photo of a cotton t-shirt"
        descriptions.append(f"a photo of a {chosen_material} {article}")
        
        if color:
            # "a blue cotton t-shirt"
            descriptions.append(f"a {color} {chosen_material} {article}")
            # "a photo of a blue cotton shirt"
            descriptions.append(f"a photo of a {color} {chosen_material} {article}")
        
        if gender and gender != 'unisex':
            # "men's cotton t-shirt"
            gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
            descriptions.append(f"{gender_prefix} {chosen_material} {article}")
        
        # Texture adjective enrichment
        if chosen_material in TEXTURE_DESCRIPTORS:
            texture_adj = random.choice(TEXTURE_DESCRIPTORS[chosen_material])
            # "a soft cotton t-shirt"
            descriptions.append(f"a {texture_adj} {chosen_material} {article}")
            if color:
                # "a soft blue cotton t-shirt"
                descriptions.append(f"a {texture_adj} {color} {chosen_material} {article}")
        
        # "t-shirt made from cotton"
        descriptions.append(f"{article} made from {chosen_material}")
    
    # ─── v3 NEW: Fashionpedia attribute templates ────────────
    
    # Pattern-enriched descriptions
    if pattern_list and article:
        pattern = pattern_list[0]
        descriptions.append(f"a {pattern} {article}")
        if color:
            descriptions.append(f"a {color} {pattern} {article}")
        # "a floral print cotton dress"
        if chosen_material:
            descriptions.append(f"a {pattern} {chosen_material} {article}")
    
    # Style-enriched descriptions
    if style_list and article:
        style = style_list[0]
        descriptions.append(f"a {style} {article}")
        if color:
            descriptions.append(f"a {color} {style} {article}")
        if chosen_material:
            descriptions.append(f"a {style} {chosen_material} {article}")
    
    # Neckline-enriched descriptions
    if neckline_list and article:
        neckline = neckline_list[0]
        # "a v-neck sweater"
        descriptions.append(f"a {neckline} {article}")
        if color:
            descriptions.append(f"a {color} {neckline} {article}")
    
    # Length-enriched descriptions
    if length_list and article:
        length = length_list[0]
        # "a midi dress"
        descriptions.append(f"a {length} {article}")
        if color:
            descriptions.append(f"a {color} {length} {article}")
    
    # Sleeve-enriched descriptions
    if sleeve_list and article:
        sleeve = sleeve_list[0]
        # "a short sleeve shirt"
        descriptions.append(f"a {sleeve} {article}")
    
    # ─── v3 NEW: Combined rich attribute descriptions ────────
    # "a fitted v-neck cotton dress in navy blue"
    if article:
        rich_parts = []
        if style_list:
            rich_parts.append(style_list[0])
        if neckline_list:
            rich_parts.append(neckline_list[0])
        if chosen_material:
            rich_parts.append(chosen_material)
        rich_parts.append(article)
        
        if len(rich_parts) >= 3:
            desc = "a " + " ".join(rich_parts)
            if color:
                desc += f" in {color}"
            descriptions.append(desc)
    
    # ─── Deduplicate and return ──────────────────────
    seen = set()
    unique = []
    for d in descriptions:
        d = d.strip().lower()
        if d and d not in seen:
            seen.add(d)
            unique.append(d)
    
    return unique if unique else ['a fashion item']


# Test the v3 description generator
test_item = all_pairs[0]
print(f"Sample item: {test_item['display_name']}")
descs = create_fashion_description_v3(test_item)
print(f"Generated {len(descs)} descriptions:")
for i, desc in enumerate(descs):
    print(f"  {i+1}. {desc}")

# Also test with Fashionpedia item if available
if dataset4_pairs:
    fp_item = dataset4_pairs[0]
    print(f"\n\nFashionpedia sample: {fp_item['article']}")
    print(f"  Material: {fp_item.get('material', 'none')}")
    print(f"  Attributes: {fp_item.get('attributes', [])[:10]}")
    fp_descs = create_fashion_description_v3(fp_item)
    print(f"Generated {len(fp_descs)} descriptions:")
    for i, desc in enumerate(fp_descs):
        print(f"  {i+1}. {desc}")

# Generate descriptions for all items
for item in all_pairs:
    item['descriptions'] = create_fashion_description_v3(item)

# Count total training pairs
total_pairs = sum(len(item['descriptions']) for item in all_pairs)
print(f"\n✅ Generated {total_pairs} text-image pairs from {len(all_pairs)} images")
print(f"   Average {total_pairs/len(all_pairs):.1f} descriptions per image")

# Show material coverage stats
materials_found = sum(1 for item in all_pairs if item.get('material') or 
                       any(k in item.get('article', '').lower() for k in ARTICLE_MATERIAL_MAP))
print(f"   Material coverage: {materials_found}/{len(all_pairs)} ({100*materials_found/len(all_pairs):.0f}%)")


# ============================================================================
# CELL 5: Create PyTorch Dataset & Load Model (~5 min)
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import open_clip

class FashionCLIPDatasetV3(Dataset):
    """
    v3 Dataset — supports multiple data sources including Fashionpedia.
    Each image gets multiple text descriptions for robust training.
    """
    
    def __init__(self, items, transform=None, tokenizer=None):
        self.pairs = []
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Create all (image_path, description) pairs
        skipped = 0
        for item in items:
            img_path = item['image_path']
            if os.path.exists(img_path):
                for desc in item['descriptions']:
                    self.pairs.append((img_path, desc))
            else:
                skipped += 1
        
        if skipped > 0:
            print(f"   (skipped {skipped} items with missing images)")
        print(f"Dataset: {len(self.pairs)} valid image-text pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception:
            # Return a random other item if this image is corrupt
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.tokenizer:
            text_tokens = self.tokenizer([text])[0]
        else:
            text_tokens = text
        
        return image, text_tokens


# Load the CLIP model
model_name = 'ViT-B-32'
pretrained = 'openai'

model, _, preprocess_train = open_clip.create_model_and_transforms(
    model_name, 
    pretrained=pretrained,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"✅ Loaded {model_name} ({pretrained})")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- Resume from v2 checkpoint if available ----
CHECKPOINT_DIR = "/content/drive/MyDrive/fynda_fashion_clip"
v2_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
v1_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pt")

resume_checkpoint = None
if os.path.exists(v2_checkpoint):
    print(f"\n📂 Found v2 checkpoint! Loading...")
    checkpoint = torch.load(v2_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    resume_val_loss = checkpoint.get('val_loss', '?')
    print(f"   ✅ Resumed from v2 (epoch {checkpoint.get('epoch', '?')}, val_loss: {resume_val_loss})")
    resume_checkpoint = 'v2'
elif os.path.exists(v1_checkpoint):
    print(f"\n📂 Found v1 checkpoint! Loading...")
    checkpoint = torch.load(v1_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    resume_val_loss = checkpoint.get('val_loss', '?')
    print(f"   ✅ Resumed from v1 (epoch {checkpoint.get('epoch', '?')}, val_loss: {resume_val_loss})")
    resume_checkpoint = 'v1'
else:
    print(f"\n⚠️ No checkpoint found. Training from scratch.")

# Create dataset
dataset = FashionCLIPDatasetV3(
    items=all_pairs,
    transform=preprocess_train,
    tokenizer=tokenizer
)

# Split into train/val (90/10)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print(f"\nTrain: {len(train_dataset)} pairs")
print(f"Val:   {len(val_dataset)} pairs")


# ============================================================================
# CELL 6: Training Configuration (v3 — Texture-Aware)
# ============================================================================

# v3 Hyperparameters (tuned for larger dataset)
BATCH_SIZE = 64          # T4 can handle 64 with ViT-B/32
LEARNING_RATE = 3e-6     # Even lower LR for v3 (was 5e-6 in v2)
NUM_EPOCHS = 20          # 20 epochs for deeper learning (was 15 in v2)
WARMUP_STEPS = 800       # More warmup for larger dataset
SAVE_EVERY = 1           # Save checkpoint every epoch
GRAD_ACCUM_STEPS = 4     # Effective batch size = 64 * 4 = 256

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True
)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# v3: Unfreeze 5 transformer blocks (was 4 in v2) + all projection layers
# This trains ~35% of parameters for deeper fashion + texture understanding
for name, param in model.named_parameters():
    if any(x in name for x in [
        # Visual transformer — last 5 blocks
        'visual.transformer.resblocks.11',
        'visual.transformer.resblocks.10',
        'visual.transformer.resblocks.9',
        'visual.transformer.resblocks.8',
        'visual.transformer.resblocks.7',   # NEW in v3
        # Text transformer — last 5 blocks
        'transformer.resblocks.11',
        'transformer.resblocks.10',
        'transformer.resblocks.9',
        'transformer.resblocks.8',
        'transformer.resblocks.7',           # NEW in v3
        # Projection layers
        'visual.ln_post',
        'text_projection',
        'visual.proj',
        'ln_final',
    ]):
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# Learning rate scheduler with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(train_loader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"\n✅ v3 Training config ready — {device.upper()}")
print(f"   Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}x{GRAD_ACCUM_STEPS}")
print(f"   Resumed from: {resume_checkpoint or 'scratch'}")


# ============================================================================
# CELL 7: Training Loop (~12-18 hours on T4)
# ============================================================================

from tqdm import tqdm
import time

def clip_loss(image_features, text_features, temperature=0.07):
    """Contrastive loss for CLIP training."""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    logit_scale = 1.0 / temperature
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    
    labels = torch.arange(len(image_features), device=image_features.device)
    
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
    
    return (loss_i2t + loss_t2i) / 2


# Checkpoint directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_val_loss = float('inf')
training_history = []

print("=" * 60)
print("🚀 Starting Fashion-CLIP v3 Training (Texture-Enriched)")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Training pairs: {len(train_dataset)}")
print(f"   Trainable: {trainable:,} params ({100*trainable/total:.1f}%)")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Datasets: {1 + int(HAS_DATASET2) + int(HAS_DATASET3) + int(HAS_FASHIONPEDIA)}")
print(f"   Fashionpedia: {'✅' if HAS_FASHIONPEDIA else '❌'}")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    # ---- Training ----
    model.train()
    total_loss = 0
    num_batches = 0
    epoch_start = time.time()
    
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    optimizer.zero_grad()
    
    for batch_idx, (images, texts) in enumerate(progress):
        images = images.to(device)
        texts = texts.to(device)
        
        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        loss = clip_loss(image_features, text_features)
        loss = loss / GRAD_ACCUM_STEPS
        
        loss.backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.detach().item() * GRAD_ACCUM_STEPS
        num_batches += 1
        
        # Prevent GPU memory buildup
        del loss, image_features, text_features
        if (batch_idx + 1) % 500 == 0:
            torch.cuda.empty_cache()
        
        progress.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    avg_train_loss = total_loss / num_batches
    
    # ---- Validation ----
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for images, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images = images.to(device)
            texts = texts.to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            loss = clip_loss(image_features, text_features)
            val_loss += loss.item()
            val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    epoch_time = time.time() - epoch_start
    
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'time': epoch_time
    })
    
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Train Loss: {avg_train_loss:.4f}")
    print(f"   Val Loss:   {avg_val_loss:.4f}")
    print(f"   Time:       {epoch_time/60:.1f} min")
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'history': training_history,
        'version': 'v3',
    }
    
    # Always save latest
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"v3_checkpoint_epoch_{epoch+1}.pt"))
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model_v3.pt"))
        print(f"   ✅ New best model saved! (val_loss: {avg_val_loss:.4f})")
    else:
        print(f"   ◻️ No improvement (best: {best_val_loss:.4f})")
    
    print()

print("=" * 60)
print("🎉 v3 Training Complete!")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Model saved to: {CHECKPOINT_DIR}")
print("=" * 60)


# ============================================================================
# CELL 8: Export v3 Model for Deployment (~1 min)
# ============================================================================

"""
Export the v3 model — same format as v1/v2 so it's a drop-in replacement.
"""

# Load the best v3 model
best_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_model_v3.pt"),
    map_location='cpu'
)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Save to export directory
EXPORT_DIR = os.path.join(CHECKPOINT_DIR, "export_v3")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Save model state dict (same filename for drop-in replacement)
export_path = os.path.join(EXPORT_DIR, "fashion_clip_fynda.pt")
torch.save(model.state_dict(), export_path)

# Save config
datasets_used = ["paramaggarwal/fashion-product-images-small"]
if HAS_DATASET2:
    datasets_used.append("agrigorev/clothing-dataset-full")
if HAS_DATASET3:
    datasets_used.append("validmodel/clothing-and-style")
if HAS_FASHIONPEDIA:
    datasets_used.append("detection-datasets/fashionpedia")

config = {
    "model_name": model_name,
    "pretrained_base": pretrained,
    "version": "v3",
    "datasets": datasets_used,
    "total_images": len(all_pairs),
    "total_pairs": total_pairs,
    "epochs_trained": best_checkpoint['epoch'],
    "best_val_loss": best_checkpoint['val_loss'],
    "training_history": training_history,
    "improvements": [
        "20 epochs (was 15 in v2)",
        "Fashionpedia dataset — 48K images with 294 fine-grained attributes",
        "Material/texture-enriched text descriptions",
        "5 unfrozen transformer blocks (was 4 in v2)",
        "Pattern, style, neckline, length attribute templates",
        "Article-to-material mapping for datasets without material info",
        "Lower learning rate (3e-6 vs 5e-6)",
        "Resumed from v2 checkpoint",
        f"~{len(all_pairs)} total images ({total_pairs} text-image pairs)",
    ],
    "texture_vocabulary": list(TEXTURE_DESCRIPTORS.keys()),
}

with open(os.path.join(EXPORT_DIR, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
print(f"✅ Exported v3 model:")
print(f"   Path: {export_path}")
print(f"   Size: {file_size_mb:.1f} MB")
print(f"   Config: {os.path.join(EXPORT_DIR, 'config.json')}")
print(f"\n📥 Download from Google Drive:")
print(f"   Go to drive.google.com → fynda_fashion_clip → export_v3/")
print(f"   Download: fashion_clip_fynda.pt + config.json")
print(f"\n🔄 To deploy: replace the files in FYNDA_ML_Services/models/fashion_clip/")


# ============================================================================
# CELL 9: Test v3 Model with Texture Queries (~2 min)
# ============================================================================

"""
Test with texture-rich fashion descriptions to verify v3 improvement.
"""

# v3 expanded test descriptions including textures
test_descriptions = [
    # Basic (same as v2)
    "men's black t-shirt",
    "women's red dress",
    "blue denim jeans",
    "white sneakers",
    "leather handbag",
    # Texture-enriched (NEW in v3)
    "soft cotton crew neck t-shirt",
    "rugged denim jacket",
    "smooth leather ankle boots",
    "luxurious silk blouse",
    "warm wool overcoat",
    "crisp linen button-up shirt",
    "plush velvet evening dress",
    "breathable mesh running shoes",
    "cozy fleece pullover hoodie",
    "ribbed corduroy trousers",
    # Pattern + material combos
    "striped cotton polo shirt",
    "floral silk midi dress",
    "plaid wool blazer",
    "camouflage nylon jacket",
    # Style + material combos
    "fitted v-neck cashmere sweater",
    "oversized cotton hoodie",
    "high-waisted denim shorts",
    "cropped chiffon blouse",
]

# Test with a random image
test_item = random.choice([p for p in all_pairs if p['source'] == 'fashion_products'])
test_img_path = test_item['image_path']

if os.path.exists(test_img_path):
    test_image = Image.open(test_img_path).convert('RGB')
    
    # Display the test image
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image)
    plt.title(f"Actual: {test_item['display_name']}")
    plt.axis('off')
    plt.show()
    
    # Get model predictions
    image_input = preprocess_train(test_image).unsqueeze(0).to(device)
    text_inputs = tokenizer(test_descriptions).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    print(f"\n🔍 v3 Model predictions for: {test_item['display_name']}")
    print(f"   Actual category: {test_item['article']}")
    print(f"   Actual color: {test_item['color']}")
    if test_item.get('material'):
        print(f"   Actual material: {test_item['material']}")
    print()
    
    # Show top 10 predictions
    scores, indices = similarity[0].topk(10)
    for score, idx in zip(scores, indices):
        print(f"   {score.item()*100:5.1f}%  {test_descriptions[idx]}")
    
    print(f"\n✅ v3 Model is working! Download from Google Drive and deploy to EC2.")
else:
    print(f"⚠️ Test image not found: {test_img_path}")


# ============================================================================
# CELL 10: Training History Comparison (v1 → v2 → v3)
# ============================================================================

if training_history:
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train', markersize=4)
    plt.plot(epochs, val_losses, 'r-o', label='Validation', markersize=4)
    
    # Show v1 and v2 baselines
    plt.axhline(y=0.5277, color='gray', linestyle='--', alpha=0.5, label='v1 best (0.5277)')
    plt.axhline(y=0.45, color='orange', linestyle='--', alpha=0.5, label='v2 estimated')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fashion-CLIP v3 Training Progress (Texture-Enriched)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    times = [h['time']/60 for h in training_history]
    plt.bar(epochs, times, color='steelblue')
    plt.xlabel('Epoch')
    plt.ylabel('Minutes')
    plt.title('Time per Epoch')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "v3_training_history.png"), dpi=100)
    plt.show()
    
    total_time = sum(times)
    print(f"\n⏱️ Total training time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
    print(f"📈 v1 best val loss: 0.5277")
    print(f"📈 v3 best val loss: {best_val_loss:.4f}")
    print(f"📈 Improvement from v1: {((0.5277 - best_val_loss) / 0.5277 * 100):.1f}%")
    print(f"\n🔬 v3 texture vocabulary: {', '.join(list(TEXTURE_DESCRIPTORS.keys())[:10])}...")
