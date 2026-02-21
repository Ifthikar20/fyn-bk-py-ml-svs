"""
=============================================================================
 Fashion-CLIP v4 Training — Lambda Cloud Edition
 ------------------------------------------------
 Adapted from the Colab v4 notebook for standalone Lambda Cloud execution.
 
 Run: python train_v4_lambda.py
 
 WHAT'S NEW IN v4:
 ✅ ViT-B-16 architecture — 4x more visual tokens than v3's ViT-B-32
 ✅ Hard negative mining — forces model to learn subtle differences
 ✅ Fashion-aware image augmentations — color jitter, crops, flips
 ✅ Layer-wise learning rate decay — deeper layers learn slower
 ✅ Cosine warmup + restart schedule — prevents premature convergence
 ✅ Attribute-aware batch construction — similar items in same batch
 ✅ Resume from v3 best checkpoint — continuous improvement
 
 Expected time: ~40-50 min on A10, ~25-30 min on A100
=============================================================================
"""

import os
import json
import random
import time
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm

# =============================================================================
# CONFIG — Adjust these paths for your Lambda instance
# =============================================================================
CHECKPOINT_DIR = os.path.expanduser("~/checkpoints")
EXPORT_DIR = os.path.expanduser("~/export")
DATA_DIR = os.path.expanduser("~/data")
FP_IMAGE_DIR = os.path.join(DATA_DIR, "fashionpedia_images")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FP_IMAGE_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 64          # A10 has 24GB — can handle larger batches than Colab T4
LEARNING_RATE = 5e-6
NUM_EPOCHS = 15
WARMUP_STEPS = 1000
GRAD_ACCUM_STEPS = 3     # Effective batch size = 64 * 3 = 192
LAYER_DECAY = 0.65
SAVE_EVERY = 1

# =============================================================================
# STEP 1: Download Datasets
# =============================================================================
print("=" * 60)
print("📥 STEP 1: Downloading Fashion Datasets")
print("=" * 60)

import kagglehub

print("\n📥 Dataset 1: Fashion Product Images (44K)...")
dataset1_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print(f"   ✅ Downloaded to: {dataset1_path}")

print("\n📥 Dataset 2: Clothing Dataset Full (5K)...")
HAS_DATASET2 = False
dataset2_path = None
try:
    dataset2_path = kagglehub.dataset_download("agrigorev/clothing-dataset-full")
    HAS_DATASET2 = True
    print(f"   ✅ Downloaded to: {dataset2_path}")
except Exception as e:
    print(f"   ⚠️ Skipped: {e}")

print("\n📥 Dataset 3: Clothes Dataset (diverse)...")
HAS_DATASET3 = False
dataset3_path = None
try:
    dataset3_path = kagglehub.dataset_download("validmodel/clothing-and-style")
    HAS_DATASET3 = True
    print(f"   ✅ Downloaded to: {dataset3_path}")
except Exception as e:
    print(f"   ⚠️ Skipped: {e}")

print("\n📥 Dataset 4: Fashionpedia (48K with attributes)...")
HAS_FASHIONPEDIA = False
fp_dataset = None
try:
    from datasets import load_dataset
    fp_dataset = load_dataset("detection-datasets/fashionpedia", split="train")
    HAS_FASHIONPEDIA = True
    print(f"   ✅ Loaded: {len(fp_dataset)} images")
except Exception as e:
    print(f"   ⚠️ Skipped: {e}")

print(f"\n{'=' * 60}")
print(f"📊 Datasets: 1 ✅ | 2 {'✅' if HAS_DATASET2 else '❌'} | 3 {'✅' if HAS_DATASET3 else '❌'} | 4 {'✅' if HAS_FASHIONPEDIA else '❌'}")
print(f"{'=' * 60}")


# =============================================================================
# STEP 2: Attribute Maps & Data Preparation
# =============================================================================
print("\n📊 STEP 2: Preparing data...")

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

# ---- Parse Dataset 1 ----
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
            'material': '', 'attributes': [], 'source': 'fashion_products',
        })

print(f"✅ Dataset 1: {len(dataset1_pairs)} images")

# ---- Parse Dataset 2 ----
dataset2_pairs = []
if HAS_DATASET2:
    for root, dirs, files in os.walk(dataset2_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                category = os.path.basename(root).lower().replace('_', ' ')
                if category and category != os.path.basename(dataset2_path).lower():
                    dataset2_pairs.append({
                        'image_path': img_path, 'color': '', 'gender': '', 'article': category,
                        'usage': '', 'season': '', 'sub_cat': '', 'master_cat': 'apparel',
                        'display_name': category, 'material': '', 'attributes': [],
                        'source': 'clothing_full',
                    })
    print(f"✅ Dataset 2: {len(dataset2_pairs)} images")

# ---- Parse Dataset 3 ----
dataset3_pairs = []
if HAS_DATASET3:
    for root, dirs, files in os.walk(dataset3_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                category = os.path.basename(root).lower().replace('_', ' ')
                if category and category != os.path.basename(dataset3_path).lower():
                    dataset3_pairs.append({
                        'image_path': img_path, 'color': '', 'gender': '', 'article': category,
                        'usage': '', 'season': '', 'sub_cat': '', 'master_cat': 'apparel',
                        'display_name': category, 'material': '', 'attributes': [],
                        'source': 'clothes_dataset',
                    })
    print(f"✅ Dataset 3: {len(dataset3_pairs)} images")

# ---- Parse Dataset 4: Fashionpedia ----
dataset4_pairs = []
if HAS_FASHIONPEDIA and fp_dataset is not None:
    print("   Parsing Fashionpedia attributes...")
    for idx, item in enumerate(fp_dataset):
        try:
            img = item.get("image")
            if img is None:
                continue
            img_path = os.path.join(FP_IMAGE_DIR, f"fp_{idx}.jpg")
            if not os.path.exists(img_path):
                img.save(img_path)

            categories = item.get("objects", {}).get("category", [])
            attributes = item.get("objects", {}).get("attribute", [])

            primary_cat = ""
            if categories:
                cat_id = categories[0] if isinstance(categories[0], int) else 0
                primary_cat = FASHIONPEDIA_CATEGORIES.get(cat_id, "").split(",")[0].strip()

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
                'image_path': img_path, 'color': '', 'gender': '', 'article': primary_cat,
                'usage': '', 'season': '', 'sub_cat': '', 'master_cat': 'apparel',
                'display_name': primary_cat,
                'material': material_names[0] if material_names else '',
                'attributes': attr_names,
                'material_list': material_names, 'pattern_list': pattern_names,
                'style_list': style_names, 'neckline_list': neckline_names,
                'length_list': length_names, 'sleeve_list': sleeve_names,
                'source': 'fashionpedia',
            })
            if (idx + 1) % 10000 == 0:
                print(f"   ... {idx + 1}/{len(fp_dataset)}")
        except Exception:
            continue
    print(f"✅ Dataset 4: {len(dataset4_pairs)} Fashionpedia images")

# Merge all
all_pairs = dataset1_pairs + dataset2_pairs + dataset3_pairs + dataset4_pairs
print(f"\n🔗 Total: {len(all_pairs)} images from {1 + int(HAS_DATASET2) + int(HAS_DATASET3) + int(HAS_FASHIONPEDIA)} datasets")


# =============================================================================
# STEP 3: Generate Text Descriptions
# =============================================================================
print("\n📝 STEP 3: Generating text descriptions...")

def create_fashion_description_v3(item):
    """Rich text description generator with material/texture enrichment."""
    color = item.get('color', '')
    gender = item.get('gender', '')
    article = item.get('article', '')
    usage = item.get('usage', '')
    season = item.get('season', '')
    sub_cat = item.get('sub_cat', '')
    display_name = item.get('display_name', '')
    material = item.get('material', '')
    pattern_list = item.get('pattern_list', [])
    style_list = item.get('style_list', [])
    neckline_list = item.get('neckline_list', [])
    length_list = item.get('length_list', [])
    sleeve_list = item.get('sleeve_list', [])

    for field in ['color', 'gender', 'article', 'usage', 'season', 'sub_cat', 'material']:
        val = item.get(field, '')
        if str(val) in ['nan', 'none', '', 'None']:
            item[field] = ''
    color, gender, article = item.get('color', ''), item.get('gender', ''), item.get('article', '')
    usage, season, sub_cat, material = item.get('usage', ''), item.get('season', ''), item.get('sub_cat', ''), item.get('material', '')

    descriptions = []

    if article:
        descriptions.append(f"a photo of a {article}")
        if color: descriptions.append(f"a photo of a {color} {article}")
        if color and gender and gender != 'unisex':
            descriptions.append(f"a photo of a {color} {gender}'s {article}")

    parts = [p for p in [color, usage, article] if p]
    if parts:
        desc = ' '.join(parts)
        if gender and gender != 'unisex': desc += f" for {gender}"
        descriptions.append(desc)

    if gender and gender != 'unisex' and article:
        gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
        parts = [p for p in [gender_prefix, color, article] if p]
        descriptions.append(' '.join(parts))

    if color and article: descriptions.append(f"{color} {article}")
    if season and color and article: descriptions.append(f"{season} {color} {article}")
    if usage and sub_cat and color: descriptions.append(f"{usage} {sub_cat} in {color}")

    if color and article:
        descriptions.append(f"a {color} {article} on a white background")
        descriptions.append(f"a closeup of a {color} {article}")

    if display_name and display_name != 'nan': descriptions.append(display_name)
    if article: descriptions.append(article)

    # Material enrichment
    chosen_material = material
    if not chosen_material and article:
        for art_key, materials in ARTICLE_MATERIAL_MAP.items():
            if art_key in article.lower():
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
            if color: descriptions.append(f"a {texture_adj} {color} {chosen_material} {article}")
        descriptions.append(f"{article} made from {chosen_material}")

    if pattern_list and article:
        p = pattern_list[0]
        descriptions.append(f"a {p} {article}")
        if color: descriptions.append(f"a {color} {p} {article}")
    if style_list and article:
        s = style_list[0]
        descriptions.append(f"a {s} {article}")
        if color: descriptions.append(f"a {color} {s} {article}")
    if neckline_list and article:
        n = neckline_list[0]
        descriptions.append(f"a {n} {article}")
        if color: descriptions.append(f"a {color} {n} {article}")
    if length_list and article:
        l = length_list[0]
        descriptions.append(f"a {l} {article}")
    if sleeve_list and article:
        descriptions.append(f"a {sleeve_list[0]} {article}")

    # Rich combined
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

    seen = set()
    unique = []
    for d in descriptions:
        d = d.strip().lower()
        if d and d not in seen:
            seen.add(d)
            unique.append(d)
    return unique if unique else ['a fashion item']


for item in all_pairs:
    item['descriptions'] = create_fashion_description_v3(item)

total_pairs = sum(len(item['descriptions']) for item in all_pairs)
print(f"✅ {total_pairs} text-image pairs from {len(all_pairs)} images ({total_pairs/len(all_pairs):.1f} per image)")


# =============================================================================
# STEP 4: Dataset & Hard Negative Sampler
# =============================================================================
print("\n⚙️ STEP 4: Building dataset & sampler...")

import open_clip

class FashionCLIPDatasetV4(Dataset):
    def __init__(self, items, transform=None, augment_transform=None, tokenizer=None, use_augmentation=True):
        self.pairs = []
        self.item_indices = []
        self.transform = transform
        self.augment_transform = augment_transform
        self.tokenizer = tokenizer
        self.use_augmentation = use_augmentation

        skipped = 0
        for item_idx, item in enumerate(items):
            if os.path.exists(item['image_path']):
                for desc in item['descriptions']:
                    self.pairs.append((item['image_path'], desc))
                    self.item_indices.append(item_idx)
            else:
                skipped += 1
        if skipped: print(f"   (skipped {skipped} missing images)")
        print(f"   Dataset: {len(self.pairs)} valid pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, text = self.pairs[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.use_augmentation and self.augment_transform and random.random() < 0.5:
                image = self.augment_transform(image)
            elif self.transform:
                image = self.transform(image)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))
        text_tokens = self.tokenizer([text])[0] if self.tokenizer else text
        return image, text_tokens


class HardNegativeBatchSampler(Sampler):
    """Groups similar items in batches to create hard negatives."""
    def __init__(self, items, pair_item_indices, batch_size=64, num_batches=None):
        self.batch_size = batch_size
        self.pair_item_indices = pair_item_indices
        self.total_pairs = len(pair_item_indices)
        self.num_batches = num_batches or (self.total_pairs // batch_size)

        self.by_article = defaultdict(list)
        self.by_color = defaultdict(list)
        self.all_indices = list(range(self.total_pairs))

        for pair_idx, item_idx in enumerate(pair_item_indices):
            item = items[item_idx]
            article = item.get('article', '').lower().strip()
            color = item.get('color', '').lower().strip()
            if article: self.by_article[article].append(pair_idx)
            if color: self.by_color[color].append(pair_idx)

        self.article_keys = [k for k, v in self.by_article.items() if len(v) >= 5]
        self.color_keys = [k for k, v in self.by_color.items() if len(v) >= 5]
        print(f"   Hard negatives: {len(self.article_keys)} article groups, {len(self.color_keys)} color groups")

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            # 40% same article
            if self.article_keys:
                pool = self.by_article[random.choice(self.article_keys)]
                batch.extend(random.sample(pool, min(int(self.batch_size * 0.4), len(pool))))
            # 30% same color
            if self.color_keys:
                pool = [i for i in self.by_color[random.choice(self.color_keys)] if i not in batch]
                batch.extend(random.sample(pool, min(int(self.batch_size * 0.3), len(pool))))
            # 30% random
            remaining = self.batch_size - len(batch)
            random_pool = [i for i in random.sample(self.all_indices, min(remaining + 20, len(self.all_indices))) if i not in batch]
            batch.extend(random_pool[:remaining])
            while len(batch) < self.batch_size:
                batch.append(random.choice(self.all_indices))
            yield batch[:self.batch_size]

    def __len__(self):
        return self.num_batches


# Fashion augmentations
fashion_augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])


# =============================================================================
# STEP 5: Load Model + Resume from Checkpoint
# =============================================================================
print("\n🧠 STEP 5: Loading ViT-B-16 model...")

model_name = 'ViT-B-16'
pretrained = 'openai'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, _, preprocess_train = open_clip.create_model_and_transforms(
    model_name, pretrained=pretrained, device=device
)
tokenizer = open_clip.get_tokenizer(model_name)

print(f"✅ {model_name} loaded on {device.upper()}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Try to resume
v3_best = os.path.join(CHECKPOINT_DIR, "best_model_v3.pt")
v4_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_v4.pt")
resume_checkpoint = None
start_epoch = 0

if os.path.exists(v4_checkpoint_path):
    print(f"\n📂 Found v4 checkpoint — resuming...")
    ckpt = torch.load(v4_checkpoint_path, map_location='cpu')
    try:
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        resume_checkpoint = 'v4'
        print(f"   ✅ Resumed from epoch {start_epoch}, val_loss: {ckpt.get('val_loss', '?')}")
    except RuntimeError as e:
        print(f"   ⚠️ Incompatible: {e}")

elif os.path.exists(v3_best):
    print(f"\n📂 Found v3 checkpoint — partial weight transfer...")
    v3_ckpt = torch.load(v3_best, map_location='cpu')
    v3_state = v3_ckpt.get('model_state_dict', v3_ckpt)
    model_state = model.state_dict()
    loaded, skipped = 0, 0
    for key, value in v3_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state)
    resume_checkpoint = 'v3_partial'
    print(f"   ✅ Loaded {loaded} layers from v3 (skipped {skipped} incompatible)")

if resume_checkpoint is None:
    print("   ⚠️ No checkpoint found — training from scratch")


# =============================================================================
# STEP 6: Build DataLoaders + Optimizer
# =============================================================================
print("\n🔧 STEP 6: Building DataLoaders...")

dataset = FashionCLIPDatasetV4(
    items=all_pairs, transform=preprocess_train,
    augment_transform=fashion_augment, tokenizer=tokenizer, use_augmentation=True,
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_dataset.dataset.use_augmentation = False
print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)}")

train_sampler = HardNegativeBatchSampler(
    items=all_pairs, pair_item_indices=dataset.item_indices[:train_size],
    batch_size=BATCH_SIZE, num_batches=train_size // BATCH_SIZE,
)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Layer-wise LR decay
param_groups = []
projection_params = []
for name, param in model.named_parameters():
    if any(x in name for x in ['visual.proj', 'text_projection', 'ln_final', 'visual.ln_post']):
        param.requires_grad = True
        projection_params.append(param)
    else:
        param.requires_grad = False

if projection_params:
    param_groups.append({'params': projection_params, 'lr': LEARNING_RATE})

for block_idx in range(11, 5, -1):
    depth = 11 - block_idx
    block_lr = LEARNING_RATE * (LAYER_DECAY ** depth)
    block_params = []
    for name, param in model.named_parameters():
        if (f'visual.transformer.resblocks.{block_idx}' in name or
            f'transformer.resblocks.{block_idx}' in name):
            param.requires_grad = True
            block_params.append(param)
    if block_params:
        param_groups.append({'params': block_params, 'lr': block_lr, 'name': f'block_{block_idx}'})

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * 5, T_mult=1, eta_min=1e-8)

model = model.to(device)


# =============================================================================
# STEP 7: Training Loop
# =============================================================================

def clip_loss_hard_negatives(image_features, text_features, temperature=0.07, hard_weight=2.0):
    """CLIP loss with hard negative emphasis."""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = 1.0 / temperature
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    batch_size = len(image_features)
    labels = torch.arange(batch_size, device=image_features.device)
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels, reduction='none')
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels, reduction='none')
    with torch.no_grad():
        mask = torch.eye(batch_size, device=image_features.device).bool()
        neg_logits = logits_per_image.masked_fill(mask, -1e9)
        hardest_neg_sim = neg_logits.max(dim=1).values
        hardness = torch.sigmoid(hardest_neg_sim)
        weights = 1.0 + (hard_weight - 1.0) * hardness
    return (weights * loss_i2t + weights * loss_t2i).mean() / 2

def standard_clip_loss(img_feat, txt_feat, temp=0.07):
    """Standard CLIP loss for fair validation comparison."""
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    logits = (1.0 / temp) * img_feat @ txt_feat.T
    labels = torch.arange(len(img_feat), device=img_feat.device)
    return (torch.nn.functional.cross_entropy(logits, labels) +
            torch.nn.functional.cross_entropy(logits.T, labels)) / 2


best_val_loss = float('inf')
training_history = []
global_step = 0

if resume_checkpoint == 'v4':
    best_val_loss = ckpt.get('val_loss', float('inf'))
    training_history = ckpt.get('history', [])

print("\n" + "=" * 60)
print("🚀 Fashion-CLIP v4 Training — Lambda Cloud")
print(f"   Model: {model_name} | Device: {device.upper()}")
print(f"   Epochs: {start_epoch + 1} → {NUM_EPOCHS}")
print(f"   Batch: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"   LR: {LEARNING_RATE} | Layer decay: {LAYER_DECAY}x")
print(f"   Pairs: {len(train_dataset)} train, {len(val_dataset)} val")
print(f"   Resume: {resume_checkpoint or 'scratch'}")
print("=" * 60 + "\n")

try:
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        optimizer.zero_grad()

        for batch_idx, (images, texts) in enumerate(progress):
            global_step += 1
            images = images.to(device)
            texts = texts.to(device)

            if global_step <= WARMUP_STEPS:
                warmup_factor = global_step / WARMUP_STEPS
                for pg in optimizer.param_groups:
                    pg['lr'] = pg.get('lr', LEARNING_RATE) * warmup_factor

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            loss = clip_loss_hard_negatives(image_features, text_features) / GRAD_ACCUM_STEPS
            loss.backward()

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

            progress.set_postfix({'loss': f'{total_loss/num_batches:.4f}',
                                  'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'})

        avg_train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, texts = images.to(device), texts.to(device)
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                val_loss += standard_clip_loss(image_features, text_features).item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start

        training_history.append({
            'epoch': epoch + 1, 'train_loss': avg_train_loss,
            'val_loss': avg_val_loss, 'time': epoch_time
        })

        print(f"\n📊 Epoch {epoch+1}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} ({epoch_time/60:.1f} min)")

        # Save checkpoint
        ckpt_data = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
            'history': training_history, 'version': 'v4', 'model_name': model_name,
        }
        torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, f"v4_epoch_{epoch+1}.pt"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, "best_model_v4.pt"))
            print(f"   ✅ New best! val_loss: {avg_val_loss:.4f}")
        else:
            print(f"   ◻️ No improvement (best: {best_val_loss:.4f})")

except KeyboardInterrupt:
    print(f"\n🛑 Interrupted — saving emergency checkpoint...")
    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': total_loss / max(num_batches, 1),
        'val_loss': best_val_loss, 'history': training_history,
        'version': 'v4', 'model_name': model_name, 'interrupted': True,
    }, os.path.join(CHECKPOINT_DIR, "v4_interrupted.pt"))
    print(f"   ✅ Saved. Best val_loss: {best_val_loss:.4f}")


# =============================================================================
# STEP 8: Export Model
# =============================================================================
print("\n" + "=" * 60)
print("📦 Exporting best model...")

best_ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_model_v4.pt"), map_location='cpu')
model.load_state_dict(best_ckpt['model_state_dict'])
model.eval()

export_path = os.path.join(EXPORT_DIR, "fashion_clip_outfi.pt")
torch.save(model.state_dict(), export_path)

config = {
    "model_name": model_name,
    "pretrained_base": pretrained,
    "version": "v4",
    "total_images": len(all_pairs),
    "total_pairs": total_pairs,
    "epochs_trained": best_ckpt['epoch'],
    "best_val_loss": best_ckpt['val_loss'],
    "training_history": training_history,
}
with open(os.path.join(EXPORT_DIR, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
print(f"✅ Exported: {export_path} ({file_size_mb:.1f} MB)")
print(f"   Config: {os.path.join(EXPORT_DIR, 'config.json')}")
print(f"   Best val_loss: {best_ckpt['val_loss']:.4f}")
print(f"\n📥 Download from Lambda:")
print(f"   scp ubuntu@<IP>:~/export/fashion_clip_outfi.pt ./")
print(f"   scp ubuntu@<IP>:~/export/config.json ./")
print(f"\n🎉 Done! Remember to STOP your Lambda instance to avoid charges.")
print("=" * 60)
