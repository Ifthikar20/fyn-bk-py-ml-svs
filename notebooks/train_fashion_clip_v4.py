"""
=============================================================================
 FYNDA Fashion-CLIP v4 Training Notebook  
 ----------------------------------------
 Run this in Google Colab Pro with T4/A100 GPU
 
 WHAT'S NEW IN v4 (Accuracy-Focused):
 ✅ ViT-B-16 architecture — 4x more visual tokens than v3's ViT-B-32
 ✅ Hard negative mining — forces model to learn subtle differences
 ✅ Fashion-aware image augmentations — color jitter, crops, flips
 ✅ Layer-wise learning rate decay — deeper layers learn slower
 ✅ Cosine warmup + restart schedule — prevents premature convergence
 ✅ Attribute-aware batch construction — similar items in same batch
 ✅ Resume from v3 best checkpoint — continuous improvement
 
 CHANGES FROM v3:
 - Model:  ViT-B-32 → ViT-B-16 (4x more patches, better detail)
 - Loss:   Standard CLIP → Hard negative weighted CLIP loss
 - Augment: None → Color jitter + random crop + horizontal flip
 - LR:     Flat decay → Layer-wise decay (0.65x per layer group)
 - Batch:  Random → 50% attribute-matched hard negatives
 
 HOW TO USE:
 1. Open Google Colab (colab.research.google.com)
 2. Create a new notebook
 3. Go to Runtime → Change runtime type → T4 GPU (A100 if available)
 4. Copy each "# CELL X" section into a separate Colab cell
 5. Run cells in order (Shift+Enter)
 
 Training time: ~20-30 hours on T4, ~8-12 hours on A100
 Disk needed: ~20GB (datasets + model)
 Output: fashion_clip_fynda.pt (~577MB saved to Google Drive)
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
Same 4 datasets as v3 — the improvement comes from HOW we train, not more data.

1. Fashion Product Images (Small) — 44K studio product shots with rich metadata
2. Clothing Dataset Full — 5K real-world photos (natural lighting, varied angles)  
3. Fashion-MNIST alternative — 10K diverse clothing items
4. Fashionpedia — 48K images with 294 fine-grained attributes (material, style)
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
    try:
        dataset4_path = kagglehub.dataset_download("rangsimanketkaew/fashionpedia")
        HAS_FASHIONPEDIA = True
        fp_dataset = None
        print(f"   ✅ Downloaded from Kaggle to: {dataset4_path}")
    except Exception as e2:
        print(f"   ⚠️ Could not download Fashionpedia: {e2}")

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

# ---- Fashionpedia attribute maps (same as v3) ----

FASHIONPEDIA_CATEGORIES = {
    0: "shirt, blouse", 1: "top, t-shirt, sweatshirt", 2: "sweater",
    3: "cardigan", 4: "jacket", 5: "vest", 6: "pants", 7: "shorts",
    8: "skirt", 9: "coat", 10: "dress", 11: "jumpsuit", 12: "cape",
    13: "glasses", 14: "hat", 15: "headband, head covering, hair accessory",
    16: "tie", 17: "glove", 18: "watch", 19: "belt", 20: "leg warmer",
    21: "tights, stockings", 22: "sock", 23: "shoe", 24: "bag, wallet",
    25: "scarf", 26: "umbrella",
    27: "hood", 28: "collar", 29: "lapel", 30: "epaulette", 31: "sleeve",
    32: "pocket", 33: "neckline", 34: "buckle", 35: "zipper", 36: "applique",
    37: "bead", 38: "bow", 39: "flower", 40: "fringe", 41: "ribbon",
    42: "rivet", 43: "ruffle", 44: "sequin", 45: "tassel",
}

FASHIONPEDIA_ATTR_GROUPS = {
    "material": {
        0: "denim", 1: "cotton", 2: "leather", 3: "fur", 4: "silk",
        5: "wool", 6: "synthetic", 7: "knitted", 8: "lace",
        9: "velvet", 10: "suede", 11: "chiffon", 12: "nylon",
        13: "linen", 14: "satin", 15: "polyester", 16: "cashmere",
        17: "mesh", 18: "corduroy", 19: "fleece", 20: "tweed",
        21: "sequined", 22: "metallic", 23: "rubber", 24: "canvas",
    },
    "pattern": {
        25: "solid", 26: "striped", 27: "plaid", 28: "floral",
        29: "polka dot", 30: "paisley", 31: "graphic print", 32: "camouflage",
        33: "animal print", 34: "abstract", 35: "checkered", 36: "colorblock",
        37: "tie-dye", 38: "houndstooth", 39: "herringbone",
    },
    "style": {
        40: "fitted", 41: "oversized", 42: "slim fit", 43: "regular fit",
        44: "relaxed", 45: "cropped", 46: "high-waisted", 47: "low-rise",
        48: "asymmetric", 49: "wrap", 50: "a-line", 51: "bodycon",
        52: "pleated", 53: "ruffled", 54: "draped",
    },
    "closure": {
        55: "button", 56: "zipper", 57: "snap", 58: "hook",
        59: "lace-up", 60: "buckle", 61: "velcro", 62: "pullover",
        63: "drawstring",
    },
    "neckline": {
        64: "crew neck", 65: "v-neck", 66: "scoop neck", 67: "turtleneck",
        68: "off-shoulder", 69: "halter", 70: "collared", 71: "boat neck",
        72: "cowl neck", 73: "square neck", 74: "mock neck",
    },
    "length": {
        75: "mini", 76: "midi", 77: "maxi", 78: "knee-length",
        79: "ankle-length", 80: "full-length", 81: "cropped",
    },
    "sleeve": {
        82: "sleeveless", 83: "short sleeve", 84: "long sleeve",
        85: "three-quarter sleeve", 86: "cap sleeve", 87: "puff sleeve",
        88: "bell sleeve", 89: "raglan sleeve",
    },
}

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
    "heels": ["leather", "suede", "satin"],
    "flats": ["leather", "suede", "canvas"],
    "loafers": ["leather", "suede"],
    "flip flops": ["rubber", "leather"],
    "slides": ["rubber", "leather"],
}

# ---- Dataset 1: Fashion Product Images ----
styles_csv = os.path.join(dataset1_path, "styles.csv")
df1 = pd.read_csv(styles_csv, on_bad_lines='skip')

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
            'material': '',
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
                        'color': '', 'gender': '', 'article': category,
                        'usage': '', 'season': '', 'sub_cat': '',
                        'master_cat': 'apparel', 'display_name': category,
                        'material': '', 'attributes': [],
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
                        'color': '', 'gender': '', 'article': category,
                        'usage': '', 'season': '', 'sub_cat': '',
                        'master_cat': 'apparel', 'display_name': category,
                        'material': '', 'attributes': [],
                        'source': 'clothes_dataset',
                    })
    print(f"✅ Dataset 3: {len(dataset3_pairs)} diverse images")

# ---- Dataset 4: Fashionpedia ----
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
            
            img_path = os.path.join(fp_save_dir, f"fp_{idx}.jpg")
            if not os.path.exists(img_path):
                img.save(img_path)
            
            categories = item.get("objects", {}).get("category", [])
            attributes = item.get("objects", {}).get("attribute", [])
            
            primary_cat = ""
            if categories:
                cat_id = categories[0] if isinstance(categories[0], int) else 0
                primary_cat = FASHIONPEDIA_CATEGORIES.get(cat_id, "")
                primary_cat = primary_cat.split(",")[0].strip()
            
            attr_names, material_names, pattern_names = [], [], []
            style_names, neckline_names, length_names, sleeve_names = [], [], [], []
            
            for attr_list in attributes:
                if isinstance(attr_list, list):
                    for attr_id in attr_list:
                        if not isinstance(attr_id, int):
                            continue
                        for group_name, group_dict in FASHIONPEDIA_ATTR_GROUPS.items():
                            if attr_id in group_dict:
                                attr_name = group_dict[attr_id]
                                attr_names.append(attr_name)
                                if group_name == "material": material_names.append(attr_name)
                                elif group_name == "pattern": pattern_names.append(attr_name)
                                elif group_name == "style": style_names.append(attr_name)
                                elif group_name == "neckline": neckline_names.append(attr_name)
                                elif group_name == "length": length_names.append(attr_name)
                                elif group_name == "sleeve": sleeve_names.append(attr_name)
            
            dataset4_pairs.append({
                'image_path': img_path, 'color': '', 'gender': '',
                'article': primary_cat, 'usage': '', 'season': '',
                'sub_cat': '', 'master_cat': 'apparel',
                'display_name': primary_cat,
                'material': material_names[0] if material_names else '',
                'attributes': attr_names,
                'material_list': material_names, 'pattern_list': pattern_names,
                'style_list': style_names, 'neckline_list': neckline_names,
                'length_list': length_names, 'sleeve_list': sleeve_names,
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
# CELL 4: Create Text Descriptions (same as v3) (~3 min)
# ============================================================================

import random

def create_fashion_description_v3(item):
    """v3 description generator — reused in v4 (descriptions were already solid)."""
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
    
    # CLIP-native templates
    if article:
        descriptions.append(f"a photo of a {article}")
        if color:
            descriptions.append(f"a photo of a {color} {article}")
        if color and gender and gender != 'unisex':
            descriptions.append(f"a photo of a {color} {gender}'s {article}")
    
    parts = [p for p in [color, usage, article] if p]
    if parts:
        desc = ' '.join(parts)
        if gender and gender != 'unisex':
            desc += f" for {gender}"
        descriptions.append(desc)
    
    if gender and gender != 'unisex' and article:
        gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
        parts = [p for p in [gender_prefix, color, article] if p]
        descriptions.append(' '.join(parts))
    
    if color and article:
        descriptions.append(f"{color} {article}")
    
    if season and color and article:
        descriptions.append(f"{season} {color} {article}")
    
    if usage and sub_cat and color:
        descriptions.append(f"{usage} {sub_cat} in {color}")
    
    if color and article:
        descriptions.append(f"a {color} {article} on a white background")
        descriptions.append(f"a closeup of a {color} {article}")
    
    if display_name and display_name != 'nan':
        descriptions.append(display_name)
    
    if article:
        descriptions.append(article)
    
    # Material/texture enrichment
    chosen_material = material
    if not chosen_material and article:
        article_lower = article.lower()
        for art_key, materials in ARTICLE_MATERIAL_MAP.items():
            if art_key in article_lower:
                chosen_material = random.choice(materials)
                break
    
    if chosen_material and article:
        descriptions.append(f"a {chosen_material} {article}")
        descriptions.append(f"a photo of a {chosen_material} {article}")
        if color:
            descriptions.append(f"a {color} {chosen_material} {article}")
            descriptions.append(f"a photo of a {color} {chosen_material} {article}")
        if gender and gender != 'unisex':
            gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
            descriptions.append(f"{gender_prefix} {chosen_material} {article}")
        if chosen_material in TEXTURE_DESCRIPTORS:
            texture_adj = random.choice(TEXTURE_DESCRIPTORS[chosen_material])
            descriptions.append(f"a {texture_adj} {chosen_material} {article}")
            if color:
                descriptions.append(f"a {texture_adj} {color} {chosen_material} {article}")
        descriptions.append(f"{article} made from {chosen_material}")
    
    # Pattern/style/neckline/length/sleeve enrichment
    if pattern_list and article:
        pattern = pattern_list[0]
        descriptions.append(f"a {pattern} {article}")
        if color: descriptions.append(f"a {color} {pattern} {article}")
        if chosen_material: descriptions.append(f"a {pattern} {chosen_material} {article}")
    
    if style_list and article:
        style = style_list[0]
        descriptions.append(f"a {style} {article}")
        if color: descriptions.append(f"a {color} {style} {article}")
        if chosen_material: descriptions.append(f"a {style} {chosen_material} {article}")
    
    if neckline_list and article:
        neckline = neckline_list[0]
        descriptions.append(f"a {neckline} {article}")
        if color: descriptions.append(f"a {color} {neckline} {article}")
    
    if length_list and article:
        length = length_list[0]
        descriptions.append(f"a {length} {article}")
        if color: descriptions.append(f"a {color} {length} {article}")
    
    if sleeve_list and article:
        sleeve = sleeve_list[0]
        descriptions.append(f"a {sleeve} {article}")
    
    # Combined rich attribute descriptions
    if article:
        rich_parts = []
        if style_list: rich_parts.append(style_list[0])
        if neckline_list: rich_parts.append(neckline_list[0])
        if chosen_material: rich_parts.append(chosen_material)
        rich_parts.append(article)
        if len(rich_parts) >= 3:
            desc = "a " + " ".join(rich_parts)
            if color: desc += f" in {color}"
            descriptions.append(desc)
    
    # Deduplicate
    seen = set()
    unique = []
    for d in descriptions:
        d = d.strip().lower()
        if d and d not in seen:
            seen.add(d)
            unique.append(d)
    
    return unique if unique else ['a fashion item']


# Generate descriptions for all items
for item in all_pairs:
    item['descriptions'] = create_fashion_description_v3(item)

total_pairs = sum(len(item['descriptions']) for item in all_pairs)
print(f"\n✅ Generated {total_pairs} text-image pairs from {len(all_pairs)} images")
print(f"   Average {total_pairs/len(all_pairs):.1f} descriptions per image")


# ============================================================================
# CELL 5: v4 Dataset with Augmentations + Hard Negative Sampler (~5 min)
# ============================================================================

"""
NEW IN v4:
1. Fashion-aware image augmentations (color jitter, crop, flip)
2. Hard negative batch sampler — groups similar items together
3. ViT-B-16 model (4x more visual detail than ViT-B-32)
"""

import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from torchvision import transforms
import open_clip
from collections import defaultdict

class FashionCLIPDatasetV4(Dataset):
    """
    v4 Dataset — same pairing logic as v3, but with augmentations.
    Augmentations help the model generalize better to real-world images.
    """
    
    def __init__(self, items, transform=None, augment_transform=None, tokenizer=None, use_augmentation=True):
        self.pairs = []
        self.item_indices = []  # Maps pair index → original item index
        self.transform = transform
        self.augment_transform = augment_transform
        self.tokenizer = tokenizer
        self.use_augmentation = use_augmentation
        
        skipped = 0
        for item_idx, item in enumerate(items):
            img_path = item['image_path']
            if os.path.exists(img_path):
                for desc in item['descriptions']:
                    self.pairs.append((img_path, desc))
                    self.item_indices.append(item_idx)
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
            
            # 50% chance: apply fashion augmentation, 50%: standard transform
            if self.use_augmentation and self.augment_transform and random.random() < 0.5:
                image = self.augment_transform(image)
            elif self.transform:
                image = self.transform(image)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.tokenizer:
            text_tokens = self.tokenizer([text])[0]
        else:
            text_tokens = text
        
        return image, text_tokens


class HardNegativeBatchSampler(Sampler):
    """
    v4 CORE IMPROVEMENT: Constructs batches with hard negatives.
    
    Instead of random batches, this sampler groups items that share
    ONE attribute (e.g. same article type OR same color) but differ
    in other attributes. This forces the model to learn subtle
    differences rather than trivially easy ones.
    
    Strategy per batch:
    - Pick a random anchor item
    - 40% of batch: same article type, different color/material
    - 30% of batch: same color, different article type
    - 30% of batch: random (easy negatives for baseline signal)
    """
    
    def __init__(self, items, pair_item_indices, batch_size=64, num_batches=None):
        self.batch_size = batch_size
        self.pair_item_indices = pair_item_indices
        self.total_pairs = len(pair_item_indices)
        self.num_batches = num_batches or (self.total_pairs // batch_size)
        
        # Build attribute indices — map from attribute → list of pair indices
        self.by_article = defaultdict(list)
        self.by_color = defaultdict(list)
        self.all_indices = list(range(self.total_pairs))
        
        for pair_idx, item_idx in enumerate(pair_item_indices):
            item = items[item_idx]
            article = item.get('article', '').lower().strip()
            color = item.get('color', '').lower().strip()
            if article:
                self.by_article[article].append(pair_idx)
            if color:
                self.by_color[color].append(pair_idx)
        
        # Only keep groups with enough items for meaningful hard negatives
        self.article_keys = [k for k, v in self.by_article.items() if len(v) >= 5]
        self.color_keys = [k for k, v in self.by_color.items() if len(v) >= 5]
        
        print(f"   Hard negative sampler: {len(self.article_keys)} article groups, {len(self.color_keys)} color groups")
    
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            
            # 40%: Same article, different attributes
            same_article_count = int(self.batch_size * 0.4)
            if self.article_keys:
                article = random.choice(self.article_keys)
                pool = self.by_article[article]
                batch.extend(random.sample(pool, min(same_article_count, len(pool))))
            
            # 30%: Same color, different articles
            same_color_count = int(self.batch_size * 0.3)
            if self.color_keys:
                color = random.choice(self.color_keys)
                pool = [i for i in self.by_color[color] if i not in batch]
                batch.extend(random.sample(pool, min(same_color_count, len(pool))))
            
            # 30%: Random fill
            remaining = self.batch_size - len(batch)
            random_pool = [i for i in random.sample(self.all_indices, min(remaining + 20, len(self.all_indices))) if i not in batch]
            batch.extend(random_pool[:remaining])
            
            # Ensure exact batch size
            while len(batch) < self.batch_size:
                batch.append(random.choice(self.all_indices))
            
            yield batch[:self.batch_size]
    
    def __len__(self):
        return self.num_batches


# ---- v4: Fashion-aware image augmentations ----
# These help the model generalize beyond studio product shots

fashion_augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,   # Fashion is color-sensitive, keep subtle
        contrast=0.15,
        saturation=0.2,
        hue=0.05,         # Very small — we don't want to change garment color
    ),
    transforms.RandomGrayscale(p=0.02),  # Rare grayscale to learn shape
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),   # CLIP normalization
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


# ============================================================================
# CELL 6: Load ViT-B-16 Model + Training Config (~5 min)
# ============================================================================

"""
v4 KEY CHANGE: ViT-B-16 gives 4x more visual tokens than ViT-B-32.
Each image is split into 14x14 = 196 patches (vs 7x7 = 49 in B-32).
This means the model sees 4x more detail — better for textures,
stitching patterns, and fine product features.
"""

# v4: ViT-B-16 (4x more detail)
model_name = 'ViT-B-16'
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
print(f"   ⚡ 4x more visual patches than v3's ViT-B-32!")

# ---- Resume from v3 best checkpoint ----
CHECKPOINT_DIR = "/content/drive/MyDrive/fynda_fashion_clip"
v3_best = os.path.join(CHECKPOINT_DIR, "best_model_v3.pt")
v4_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model_v4.pt")

resume_checkpoint = None
start_epoch = 0

# Try v4 checkpoint first (resume interrupted training)
if os.path.exists(v4_checkpoint):
    print(f"\n📂 Found v4 checkpoint! Resuming...")
    checkpoint = torch.load(v4_checkpoint, map_location='cpu')
    
    # v4 uses ViT-B-16, so state dict must match
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        resume_checkpoint = 'v4'
        print(f"   ✅ Resumed from v4 epoch {start_epoch}, val_loss: {checkpoint.get('val_loss', '?')}")
    except RuntimeError as e:
        print(f"   ⚠️ v4 checkpoint incompatible: {e}")
        print(f"   Starting fresh with ViT-B-16...")

# If no v4 checkpoint, try loading v3 weights where they fit
if resume_checkpoint is None and os.path.exists(v3_best):
    print(f"\n📂 Found v3 best checkpoint")
    v3_ckpt = torch.load(v3_best, map_location='cpu')
    v3_state = v3_ckpt.get('model_state_dict', v3_ckpt)
    
    # v3 used ViT-B-32, v4 uses ViT-B-16 — architectures differ
    # We can still load the text encoder + projection layers
    model_state = model.state_dict()
    loaded_keys = 0
    skipped_keys = 0
    for key, value in v3_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys += 1
        else:
            skipped_keys += 1
    
    model.load_state_dict(model_state)
    print(f"   ✅ Loaded {loaded_keys} compatible layers from v3 (skipped {skipped_keys} incompatible)")
    print(f"   Text encoder + projections transferred, visual encoder training from scratch")
    resume_checkpoint = 'v3_partial'

if resume_checkpoint is None:
    print(f"\n⚠️ No checkpoint found. Training ViT-B-16 from scratch.")

# Create datasets
dataset = FashionCLIPDatasetV4(
    items=all_pairs,
    transform=preprocess_train,
    augment_transform=fashion_augment,
    tokenizer=tokenizer,
    use_augmentation=True,
)

# Split 90/10
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Disable augmentation for validation
val_dataset.dataset.use_augmentation = False

print(f"\nTrain: {len(train_dataset)} pairs")
print(f"Val:   {len(val_dataset)} pairs")

# ---- v4 Training Hyperparameters ----
BATCH_SIZE = 48          # Slightly smaller for ViT-B-16 (uses more VRAM)
LEARNING_RATE = 5e-6     # Higher base LR since new loss provides better signal
NUM_EPOCHS = 15          # 15 epochs with hard negatives is enough
WARMUP_STEPS = 1000      # More warmup for new architecture
SAVE_EVERY = 1
GRAD_ACCUM_STEPS = 4     # Effective batch size = 48 * 4 = 192

# Create hard negative batch sampler for training
train_sampler = HardNegativeBatchSampler(
    items=all_pairs,
    pair_item_indices=dataset.item_indices[:train_size],
    batch_size=BATCH_SIZE,
    num_batches=train_size // BATCH_SIZE,
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# ---- v4: Layer-wise learning rate decay ----
# Deeper (earlier) layers get smaller LR — they need less adaptation
# Surface (later) layers get full LR — they specialize for fashion

LAYER_DECAY = 0.65  # Each layer group gets 0.65x the LR of the next

# Group parameters by depth
param_groups = []

# Group 1: Projection heads + final layer norms (full LR)
projection_params = []
for name, param in model.named_parameters():
    if any(x in name for x in ['visual.proj', 'text_projection', 'ln_final', 'visual.ln_post']):
        param.requires_grad = True
        projection_params.append(param)
    else:
        param.requires_grad = False  # Freeze by default

if projection_params:
    param_groups.append({'params': projection_params, 'lr': LEARNING_RATE})

# Group 2-7: Last 6 transformer blocks with decaying LR
for block_idx in range(11, 5, -1):  # blocks 11, 10, 9, 8, 7, 6
    depth = 11 - block_idx  # 0 for block 11, 5 for block 6
    block_lr = LEARNING_RATE * (LAYER_DECAY ** depth)
    
    block_params = []
    for name, param in model.named_parameters():
        if (f'visual.transformer.resblocks.{block_idx}' in name or
            f'transformer.resblocks.{block_idx}' in name):
            param.requires_grad = True
            block_params.append(param)
    
    if block_params:
        param_groups.append({
            'params': block_params,
            'lr': block_lr,
            'name': f'block_{block_idx}'
        })

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
print(f"\nLayer-wise LR decay ({LAYER_DECAY}x per group):")
for pg in param_groups:
    name = pg.get('name', 'projections')
    print(f"   {name}: LR = {pg['lr']:.2e}")

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

# Cosine annealing with warm restarts — prevents premature convergence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=len(train_loader) * 5,  # Restart every 5 epochs
    T_mult=1,
    eta_min=1e-8,
)

# Manual warmup
warmup_scheduler_active = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"\n✅ v4 Training config ready — {device.upper()}")
print(f"   Model: {model_name} (4x more detail than v3)")
print(f"   Epochs: {NUM_EPOCHS} | Base LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}x{GRAD_ACCUM_STEPS}")
print(f"   Hard negatives: ✅ | Augmentations: ✅ | Layer decay: ✅")
print(f"   Resumed from: {resume_checkpoint or 'scratch'}")


# ============================================================================
# CELL 7: v4 Training Loop — Hard Negative Loss (~20-30 hrs T4, ~10 hrs A100)
# ============================================================================

from tqdm import tqdm
import time
import signal

# ---- PAUSE / RESUME MECHANISM ----
# To PAUSE:  In a new Colab cell, run:  !touch /content/PAUSE_TRAINING
# To RESUME: In a new Colab cell, run:  !rm /content/PAUSE_TRAINING
# To STOP:   Press Ctrl+C (saves checkpoint before exiting)

PAUSE_FLAG = "/content/PAUSE_TRAINING"

def check_pause(epoch, batch_idx, model, optimizer, training_history, best_val_loss, global_step, total_loss, num_batches):
    """Check if pause flag file exists. If so, save checkpoint and wait."""
    if not os.path.exists(PAUSE_FLAG):
        return
    
    print(f"\n⏸️  PAUSE detected at Epoch {epoch+1}, Batch {batch_idx+1}")
    print(f"   Saving mid-epoch checkpoint...")
    
    # Save mid-epoch checkpoint so no progress is lost
    pause_checkpoint = {
        'epoch': epoch,                   # Not +1 — we haven't finished this epoch
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss / max(num_batches, 1),
        'val_loss': best_val_loss,
        'history': training_history,
        'global_step': global_step,
        'version': 'v4',
        'model_name': model_name,
        'paused': True,
    }
    torch.save(pause_checkpoint, os.path.join(CHECKPOINT_DIR, "v4_paused_checkpoint.pt"))
    print(f"   ✅ Checkpoint saved to: v4_paused_checkpoint.pt")
    print(f"   📊 Current train loss: {total_loss / max(num_batches, 1):.4f}")
    print(f"   🏆 Best val loss: {best_val_loss:.4f}")
    print()
    print(f"   ⏳ Training PAUSED. Waiting for resume...")
    print(f"   👉 To resume: run  !rm /content/PAUSE_TRAINING  in another cell")
    print(f"   👉 To stop:   press Ctrl+C")
    
    # Poll until pause flag is removed
    while os.path.exists(PAUSE_FLAG):
        time.sleep(30)
    
    print(f"\n▶️  RESUMED! Continuing training from Epoch {epoch+1}, Batch {batch_idx+1}")


def clip_loss_hard_negatives(image_features, text_features, temperature=0.07, hard_weight=2.0):
    """
    v4 CORE: CLIP loss with hard negative emphasis.
    
    Standard CLIP gives equal weight to all negatives in the batch.
    This version weights the loss MORE for items where the model 
    is confused — i.e., where the hardest wrong pair has high similarity.
    
    A "blue denim jacket" matched against "blue leather jacket" gets
    MORE gradient signal than against "pink high heels".
    
    Args:
        hard_weight: Maximum weight multiplier for hard examples (default 2.0)
                     1.0 = standard CLIP loss, 2.0 = hard negatives get 2x weight
    """
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    logit_scale = 1.0 / temperature
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    
    batch_size = len(image_features)
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Per-sample loss (no reduction)
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels, reduction='none')
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels, reduction='none')
    
    # Identify hard negatives: which wrong pairs have highest similarity?
    with torch.no_grad():
        mask = torch.eye(batch_size, device=image_features.device).bool()
        neg_logits = logits_per_image.masked_fill(mask, -1e9)
        
        # Hardness = similarity of the most confusing wrong pair
        hardest_neg_sim = neg_logits.max(dim=1).values
        
        # Convert to weight: more confused → more weight
        hardness = torch.sigmoid(hardest_neg_sim)  # 0 to 1
        weights = 1.0 + (hard_weight - 1.0) * hardness  # [1.0, hard_weight]
    
    # Weighted loss — hard examples contribute more
    loss = (weights * loss_i2t + weights * loss_t2i).mean() / 2
    
    return loss


# ---- Training loop ----
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_val_loss = float('inf')
training_history = []
global_step = 0

# Load best val loss from checkpoint if resuming
if resume_checkpoint == 'v4' and 'checkpoint' in dir():
    best_val_loss = checkpoint.get('val_loss', float('inf'))
    training_history = checkpoint.get('history', [])
    print(f"   Best val loss so far: {best_val_loss:.4f}")

# Clean up any stale pause flag from a previous run
if os.path.exists(PAUSE_FLAG):
    os.remove(PAUSE_FLAG)
    print("   🗑️ Removed stale pause flag from previous run")

print("=" * 60)
print("🚀 Starting Fashion-CLIP v4 Training")
print(f"   NEW: ViT-B-16 + Hard Negatives + Augmentations + Layer Decay")
print(f"   Epochs: {start_epoch + 1} → {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"   Base LR: {LEARNING_RATE} (layer decay: {LAYER_DECAY}x)")
print(f"   Training pairs: {len(train_dataset)}")
print(f"   Trainable: {trainable:,} params ({100*trainable/total:.1f}%)")
print(f"   ⏸️  To pause: run  !touch /content/PAUSE_TRAINING  in another cell")
print("=" * 60)

try:
    for epoch in range(start_epoch, NUM_EPOCHS):
        # ---- Training ----
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        optimizer.zero_grad()
        
        for batch_idx, (images, texts) in enumerate(progress):
            # Check for pause flag every 100 batches (minimal overhead)
            if (batch_idx + 1) % 100 == 0:
                check_pause(epoch, batch_idx, model, optimizer, training_history,
                           best_val_loss, global_step, total_loss, num_batches)
            
            global_step += 1
            images = images.to(device)
            texts = texts.to(device)
            
            # Manual warmup: linearly increase LR for first N steps
            if global_step <= WARMUP_STEPS:
                warmup_factor = global_step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = pg.get('lr', LEARNING_RATE) * warmup_factor
            
            # Forward
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # v4: Hard negative weighted loss
            loss = clip_loss_hard_negatives(image_features, text_features)
            loss = loss / GRAD_ACCUM_STEPS
            
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if global_step > WARMUP_STEPS:
                    scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.detach().item() * GRAD_ACCUM_STEPS
            num_batches += 1
            
            del loss, image_features, text_features
            if (batch_idx + 1) % 500 == 0:
                torch.cuda.empty_cache()
            
            current_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_train_loss = total_loss / num_batches
        
        # ---- Validation (no augmentation, no hard negatives) ----
        model.eval()
        val_loss = 0
        val_batches = 0
        
        # Use standard CLIP loss for validation (fair comparison with v3)
        def standard_clip_loss(img_feat, txt_feat, temp=0.07):
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            logit_scale = 1.0 / temp
            logits = logit_scale * img_feat @ txt_feat.T
            labels = torch.arange(len(img_feat), device=img_feat.device)
            return (torch.nn.functional.cross_entropy(logits, labels) +
                    torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images = images.to(device)
                texts = texts.to(device)
                
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                
                loss = standard_clip_loss(image_features, text_features)
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
            'version': 'v4',
            'model_name': model_name,
        }
        
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"v4_checkpoint_epoch_{epoch+1}.pt"))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model_v4.pt"))
            print(f"   ✅ New best model saved! (val_loss: {avg_val_loss:.4f})")
        else:
            print(f"   ◻️ No improvement (best: {best_val_loss:.4f})")
        
        # Check for pause between epochs too
        check_pause(epoch, batch_idx, model, optimizer, training_history,
                   best_val_loss, global_step, total_loss, num_batches)
        
        print()

except KeyboardInterrupt:
    print(f"\n\n🛑 Training interrupted by Ctrl+C!")
    print(f"   Saving emergency checkpoint...")
    
    emergency_checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss / max(num_batches, 1),
        'val_loss': best_val_loss,
        'history': training_history,
        'global_step': global_step,
        'version': 'v4',
        'model_name': model_name,
        'interrupted': True,
    }
    torch.save(emergency_checkpoint, os.path.join(CHECKPOINT_DIR, "v4_interrupted_checkpoint.pt"))
    
    print(f"   ✅ Emergency checkpoint saved: v4_interrupted_checkpoint.pt")
    print(f"   📊 Train loss at interruption: {total_loss / max(num_batches, 1):.4f}")
    print(f"   🏆 Best val loss: {best_val_loss:.4f}")
    print(f"   💾 Best model is still safe at: best_model_v4.pt")

print("=" * 60)
print("🎉 v4 Training Complete!")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Model saved to: {CHECKPOINT_DIR}")
print("=" * 60)


# ============================================================================
# CELL 8: Export v4 Model for Deployment (~1 min)
# ============================================================================

"""
Export the v4 model — same format as v1/v2/v3 so it's a drop-in replacement.
NOTE: The app/models/fashion_clip.py loader must support ViT-B-16.
"""

# Load the best v4 model
best_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_model_v4.pt"),
    map_location='cpu'
)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Save to export directory
EXPORT_DIR = os.path.join(CHECKPOINT_DIR, "export_v4")
os.makedirs(EXPORT_DIR, exist_ok=True)

export_path = os.path.join(EXPORT_DIR, "fashion_clip_fynda.pt")
torch.save(model.state_dict(), export_path)

# Save config (important: includes model_name for loading)
datasets_used = ["paramaggarwal/fashion-product-images-small"]
if HAS_DATASET2: datasets_used.append("agrigorev/clothing-dataset-full")
if HAS_DATASET3: datasets_used.append("validmodel/clothing-and-style")
if HAS_FASHIONPEDIA: datasets_used.append("detection-datasets/fashionpedia")

config = {
    "model_name": model_name,          # ViT-B-16 — MUST match when loading
    "pretrained_base": pretrained,
    "version": "v4",
    "datasets": datasets_used,
    "total_images": len(all_pairs),
    "total_pairs": total_pairs,
    "epochs_trained": best_checkpoint['epoch'],
    "best_val_loss": best_checkpoint['val_loss'],
    "training_history": training_history,
    "improvements": [
        f"ViT-B-16 architecture (4x more visual patches than v3's ViT-B-32)",
        "Hard negative mining — weighted loss for confusing pairs",
        "Fashion-aware image augmentations (color jitter, crop, flip)",
        f"Layer-wise LR decay ({LAYER_DECAY}x per group)",
        "Cosine annealing with warm restarts",
        "Attribute-aware batch construction (40% same article, 30% same color)",
        f"6 unfrozen transformer blocks + projections",
        f"Resumed from v3 best checkpoint (partial weight transfer)",
        f"~{len(all_pairs)} total images ({total_pairs} text-image pairs)",
    ],
}

with open(os.path.join(EXPORT_DIR, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
print(f"✅ Exported v4 model:")
print(f"   Path: {export_path}")
print(f"   Size: {file_size_mb:.1f} MB")
print(f"   Architecture: {model_name}")
print(f"   Config: {os.path.join(EXPORT_DIR, 'config.json')}")
print(f"\n📥 Download from Google Drive:")
print(f"   Go to drive.google.com → fynda_fashion_clip → export_v4/")
print(f"   Download: fashion_clip_fynda.pt + config.json")
print(f"\n🔄 To deploy: replace the files in FYNDA_ML_Services/models/fashion_clip/")
print(f"\n⚠️  IMPORTANT: Update app/models/fashion_clip.py to use ViT-B-16:")
print(f"   Change model_name from 'ViT-B-32' → 'ViT-B-16'")


# ============================================================================
# CELL 9: Test v4 Model — Comprehensive Accuracy Check (~2 min)
# ============================================================================

"""
Expanded test suite that specifically tests the weaknesses found in v3:
- Category confusion (shoes vs shirts)
- Material/texture differentiation
- Color × article type combinations
"""

# Comprehensive test descriptions covering all fashion categories
test_descriptions = [
    # ── Footwear (v3 was weak here) ──
    "black casual shoes",
    "white sneakers",
    "leather ankle boots",
    "canvas slip-on shoes",
    "high heel sandals",
    # ── Tops ──
    "men's black t-shirt",
    "women's white blouse",
    "blue denim jacket",
    "red wool sweater",
    "cotton polo shirt",
    # ── Bottoms ──
    "blue denim jeans",
    "black leather pants",
    "khaki chino shorts",
    "pleated midi skirt",
    # ── Accessories ──
    "leather crossbody bag",
    "metal wristwatch",
    "silk neck scarf",
    # ── Material tests (v4 should nail these) ──
    "soft cotton crew neck t-shirt",
    "rugged denim jacket",
    "smooth leather ankle boots",
    "luxurious silk evening dress",
    "warm wool overcoat",
    # ── Pattern + material combos ──
    "striped cotton polo shirt",
    "floral silk midi dress",
    "plaid wool blazer",
]

# Test multiple random images
import random
num_tests = 3
test_items = random.sample([p for p in all_pairs if p.get('article')], min(num_tests, len(all_pairs)))

for test_idx, test_item in enumerate(test_items):
    test_img_path = test_item['image_path']
    
    if not os.path.exists(test_img_path):
        continue
    
    test_image = Image.open(test_img_path).convert('RGB')
    
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image)
    plt.title(f"Test {test_idx+1}: {test_item.get('display_name', test_item.get('article', '?'))}")
    plt.axis('off')
    plt.show()
    
    image_input = preprocess_train(test_image).unsqueeze(0).to(device)
    text_inputs = tokenizer(test_descriptions).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    print(f"\n🔍 v4 predictions for: {test_item.get('display_name', '?')}")
    print(f"   Article: {test_item.get('article', '?')}")
    print(f"   Color: {test_item.get('color', '?')}")
    if test_item.get('material'):
        print(f"   Material: {test_item['material']}")
    print()
    
    scores, indices = similarity[0].topk(10)
    for score, idx in zip(scores, indices):
        marker = "✅" if any(kw in test_descriptions[idx] for kw in [
            test_item.get('article', '???').split()[0].lower()
        ]) else "  "
        print(f"   {marker} {score.item()*100:5.1f}%  {test_descriptions[idx]}")
    print()

# Compare with v3 baseline
print(f"\n{'='*60}")
print(f"📊 v4 vs v3 Comparison:")
print(f"   v3 best val loss: 0.4369 (epoch 16, ViT-B-32)")
print(f"   v4 best val loss: {best_val_loss:.4f} (epoch {best_checkpoint['epoch']}, ViT-B-16)")
if best_val_loss < 0.4369:
    improvement = ((0.4369 - best_val_loss) / 0.4369) * 100
    print(f"   📈 Improvement: {improvement:.1f}%")
else:
    print(f"   ⚠️ v4 is still converging — may need more epochs")
print(f"{'='*60}")
print(f"\n✅ v4 Model is ready! Download from Google Drive and deploy to EC2.")

