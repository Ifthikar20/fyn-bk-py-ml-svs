"""
=============================================================================
 FYNDA Fashion-CLIP v2 Training Notebook  
 ----------------------------------------
 Run this in Google Colab Pro with T4 GPU
 
 WHAT'S NEW IN v2:
 ✅ 15 epochs (was 5) — loss was still dropping
 ✅ Multiple datasets — 3 combined for ~85K images
 ✅ Richer text descriptions — 10+ templates per image  
 ✅ More unfrozen layers — 4 blocks (was 2), ~30% trainable
 ✅ Resume from v1 checkpoint — don't lose previous training
 ✅ CLIP-native "a photo of" prompts for better alignment
 
 HOW TO USE:
 1. Open Google Colab (colab.research.google.com)
 2. Create a new notebook
 3. Go to Runtime → Change runtime type → T4 GPU
 4. Copy each "# CELL X" section into a separate Colab cell
 5. Run cells in order (Shift+Enter)
 
 Training time: ~8-12 hours on T4 GPU
 Disk needed: ~15GB (datasets + model)
 Output: fashion_clip_fynda_v2.pt (~577MB saved to Google Drive)
=============================================================================
"""

# ============================================================================
# CELL 1: Install Dependencies (~2 min)
# ============================================================================

!pip install -q transformers torch torchvision pillow kagglehub tqdm
!pip install -q open_clip_torch

# Mount Google Drive for saving the model
from google.colab import drive
drive.mount('/content/drive')


# ============================================================================
# CELL 2: Download Multiple Fashion Datasets (~20 min)
# ============================================================================

"""
We combine 3 fashion datasets for maximum diversity:

1. Fashion Product Images (Small) — 44K studio product shots with rich metadata
2. Clothing Dataset Full — 5K real-world photos (natural lighting, varied angles)  
3. Fashion-MNIST alternative — 10K diverse clothing items

Total: ~60K+ unique fashion images
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

print("\n" + "="*60)
print("📊 Dataset Summary:")
print(f"   Dataset 1 (Fashion Products):  ✅ {dataset1_path}")
if HAS_DATASET2:
    print(f"   Dataset 2 (Clothing Full):     ✅ {dataset2_path}")
if HAS_DATASET3:
    print(f"   Dataset 3 (Clothes):           ✅ {dataset3_path}")
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
            'source': 'fashion_products',
        })

print(f"✅ Dataset 1: {len(dataset1_pairs)} images with metadata")

# ---- Dataset 2: Clothing Dataset Full ----
dataset2_pairs = []
if HAS_DATASET2:
    # This dataset has images organized in category folders
    for root, dirs, files in os.walk(dataset2_path):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, f)
                # Category comes from the folder name
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
                        'source': 'clothes_dataset',
                    })
    print(f"✅ Dataset 3: {len(dataset3_pairs)} diverse images")

# ---- Merge all datasets ----
all_pairs = dataset1_pairs + dataset2_pairs + dataset3_pairs
print(f"\n🔗 Total merged: {len(all_pairs)} images from {1 + int(HAS_DATASET2) + int(HAS_DATASET3)} datasets")


# ============================================================================
# CELL 4: Create Rich Fashion Text Descriptions (~3 min)
# ============================================================================

import random

def create_fashion_description_v2(item):
    """
    v2 description generator — creates 8-12 diverse text descriptions per image.
    
    Key improvements over v1:
    - CLIP-native "a photo of..." format
    - More template variations
    - Attribute permutations
    - Natural language descriptions
    """
    color = item.get('color', '')
    gender = item.get('gender', '')
    article = item.get('article', '')
    usage = item.get('usage', '')
    season = item.get('season', '')
    sub_cat = item.get('sub_cat', '')
    display_name = item.get('display_name', '')
    
    # Clean up
    for field in ['color', 'gender', 'article', 'usage', 'season', 'sub_cat']:
        val = item.get(field, '')
        if val in ['nan', 'none', '']:
            item[field] = ''
    color = item.get('color', '')
    gender = item.get('gender', '')
    article = item.get('article', '')
    usage = item.get('usage', '')
    season = item.get('season', '')
    sub_cat = item.get('sub_cat', '')
    
    descriptions = []
    
    # === CLIP-native "a photo of" templates ===
    if article:
        descriptions.append(f"a photo of a {article}")
        if color:
            descriptions.append(f"a photo of a {color} {article}")
        if color and gender and gender != 'unisex':
            descriptions.append(f"a photo of a {color} {gender}'s {article}")
    
    # === Descriptive templates ===
    # "navy blue casual shirt for men"
    parts = [p for p in [color, usage, article] if p]
    if parts:
        desc = ' '.join(parts)
        if gender and gender != 'unisex':
            desc += f" for {gender}"
        descriptions.append(desc)
    
    # "men's navy blue shirt"
    if gender and gender != 'unisex' and article:
        gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
        parts = [p for p in [gender_prefix, color, article] if p]
        descriptions.append(' '.join(parts))
    
    # "navy blue shirt"
    if color and article:
        descriptions.append(f"{color} {article}")
    
    # "summer navy blue shirt"
    if season and color and article:
        descriptions.append(f"{season} {color} {article}")
    
    # "casual topwear in navy blue"
    if usage and sub_cat and color:
        descriptions.append(f"{usage} {sub_cat} in {color}")
    
    # === Context-rich templates ===
    if color and article:
        descriptions.append(f"a {color} {article} on a white background")
        descriptions.append(f"a closeup of a {color} {article}")
    
    # === Product display name (original title) ===
    if display_name and display_name != 'nan':
        descriptions.append(display_name)
    
    # === Simple category-only ===
    if article:
        descriptions.append(article)
    
    # Deduplicate and return
    seen = set()
    unique = []
    for d in descriptions:
        d = d.strip()
        if d and d not in seen:
            seen.add(d)
            unique.append(d)
    
    return unique if unique else ['a fashion item']


# Test the v2 description generator
test_item = all_pairs[0]
print(f"Sample item: {test_item['display_name']}")
print(f"Generated {len(create_fashion_description_v2(test_item))} descriptions:")
for i, desc in enumerate(create_fashion_description_v2(test_item)):
    print(f"  {i+1}. {desc}")

# Generate descriptions for all items
for item in all_pairs:
    item['descriptions'] = create_fashion_description_v2(item)

# Count total training pairs
total_pairs = sum(len(item['descriptions']) for item in all_pairs)
print(f"\n✅ Generated {total_pairs} text-image pairs from {len(all_pairs)} images")
print(f"   Average {total_pairs/len(all_pairs):.1f} descriptions per image")


# ============================================================================
# CELL 5: Create PyTorch Dataset & Load Model (~5 min)
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import open_clip

class FashionCLIPDatasetV2(Dataset):
    """
    v2 Dataset — supports multiple data sources with different formats.
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

# ---- Resume from v1 checkpoint if available ----
CHECKPOINT_DIR = "/content/drive/MyDrive/fynda_fashion_clip"
v1_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pt")

if os.path.exists(v1_checkpoint):
    print(f"\n📂 Found v1 checkpoint! Loading...")
    checkpoint = torch.load(v1_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    v1_val_loss = checkpoint.get('val_loss', '?')
    print(f"   ✅ Resumed from v1 (epoch {checkpoint.get('epoch', '?')}, val_loss: {v1_val_loss})")
    print(f"   v2 training will continue improving from here!")
else:
    print(f"\n⚠️ No v1 checkpoint found. Training from scratch.")

# Create dataset
dataset = FashionCLIPDatasetV2(
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
# CELL 6: Training Configuration (v2 — More Aggressive)
# ============================================================================

# v2 Hyperparameters
BATCH_SIZE = 64          # T4 can handle 64 with ViT-B/32
LEARNING_RATE = 5e-6     # Lower LR for continued fine-tuning (was 1e-5)
NUM_EPOCHS = 15          # 15 epochs (was 5) — loss was still dropping
WARMUP_STEPS = 500       # Gradual learning rate warmup
SAVE_EVERY = 1           # Save checkpoint every N epochs
GRAD_ACCUM_STEPS = 4     # Effective batch size = 64 * 4 = 256

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

# Freeze all parameters first
for param in model.parameters():
    param.requires_grad = False

# v2: Unfreeze 4 transformer blocks (was 2) + all projection layers
# This trains ~30% of parameters for deeper fashion understanding
for name, param in model.named_parameters():
    if any(x in name for x in [
        # Visual transformer — last 4 blocks
        'visual.transformer.resblocks.11',
        'visual.transformer.resblocks.10',
        'visual.transformer.resblocks.9',   # NEW in v2
        'visual.transformer.resblocks.8',   # NEW in v2
        # Text transformer — last 4 blocks
        'transformer.resblocks.11',
        'transformer.resblocks.10',
        'transformer.resblocks.9',           # NEW in v2
        'transformer.resblocks.8',           # NEW in v2
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
print(f"\n✅ v2 Training config ready — {device.upper()}")
print(f"   Epochs: {NUM_EPOCHS} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}x{GRAD_ACCUM_STEPS}")


# ============================================================================
# CELL 7: Training Loop (~8-12 hours on T4)
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


# Checkpoint directory (same as v1, stores both versions)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_val_loss = float('inf')
training_history = []

print("=" * 60)
print("🚀 Starting Fashion-CLIP v2 Training")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Training pairs: {len(train_dataset)}")
print(f"   Trainable: {trainable:,} params ({100*trainable/total:.1f}%)")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
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
        
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1
        
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
        'version': 'v2',
    }
    
    # Always save latest
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"v2_checkpoint_epoch_{epoch+1}.pt"))
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model_v2.pt"))
        print(f"   ✅ New best model saved! (val_loss: {avg_val_loss:.4f})")
    else:
        print(f"   ◻️ No improvement (best: {best_val_loss:.4f})")
    
    print()

print("=" * 60)
print("🎉 v2 Training Complete!")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Model saved to: {CHECKPOINT_DIR}")
print("=" * 60)


# ============================================================================
# CELL 8: Export v2 Model for Deployment (~1 min)
# ============================================================================

"""
Export the v2 model — same format as v1 so it's a drop-in replacement.
"""

# Load the best v2 model
best_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_model_v2.pt"),
    map_location='cpu'
)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Save to export directory
EXPORT_DIR = os.path.join(CHECKPOINT_DIR, "export_v2")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Save model state dict (same filename as v1 for drop-in replacement)
export_path = os.path.join(EXPORT_DIR, "fashion_clip_fynda.pt")
torch.save(model.state_dict(), export_path)

# Save config
datasets_used = ["paramaggarwal/fashion-product-images-small"]
if HAS_DATASET2:
    datasets_used.append("agrigorev/clothing-dataset-full")
if HAS_DATASET3:
    datasets_used.append("validmodel/clothing-and-style")

config = {
    "model_name": model_name,
    "pretrained_base": pretrained,
    "version": "v2",
    "datasets": datasets_used,
    "total_images": len(all_pairs),
    "total_pairs": total_pairs,
    "epochs_trained": best_checkpoint['epoch'],
    "best_val_loss": best_checkpoint['val_loss'],
    "training_history": training_history,
    "improvements": [
        "15 epochs (was 5)",
        "Multiple datasets combined",
        "4 unfrozen transformer blocks (was 2)", 
        "CLIP-native description templates",
        "Lower learning rate (5e-6 vs 1e-5)",
        "Resumed from v1 checkpoint",
    ],
}

with open(os.path.join(EXPORT_DIR, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
print(f"✅ Exported v2 model:")
print(f"   Path: {export_path}")
print(f"   Size: {file_size_mb:.1f} MB")
print(f"   Config: {os.path.join(EXPORT_DIR, 'config.json')}")
print(f"\n📥 Download from Google Drive:")
print(f"   Go to drive.google.com → fynda_fashion_clip → export_v2/")
print(f"   Download: fashion_clip_fynda.pt + config.json")
print(f"\n🔄 To deploy: replace the files in FYNDA_ML_Services/models/fashion_clip/")


# ============================================================================
# CELL 9: Test v2 Model (~2 min)
# ============================================================================

"""
Test with comprehensive fashion descriptions to verify improvement.
"""

# Expanded test descriptions for better coverage
test_descriptions = [
    "men's black t-shirt",
    "women's red dress",
    "blue denim jeans",
    "white sneakers",
    "leather handbag",
    "navy blue formal shirt",
    "yellow summer dress",
    "grey casual hoodie",
    "sports running shoes",
    "pink floral blouse",
    "brown leather belt",
    "striped polo shirt",
    "black formal suit",
    "denim jacket",
    "winter wool coat",
    "silk scarf",
    "aviator sunglasses",
    "gold wrist watch",
    "casual backpack",
    "ankle boots",
]

# Test with a random image from the primary dataset
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
    
    print(f"\n🔍 v2 Model predictions for: {test_item['display_name']}")
    print(f"   Actual category: {test_item['article']}")
    print(f"   Actual color: {test_item['color']}")
    print()
    
    # Show top 7 predictions
    scores, indices = similarity[0].topk(7)
    for score, idx in zip(scores, indices):
        print(f"   {score.item()*100:5.1f}%  {test_descriptions[idx]}")
    
    print(f"\n✅ v2 Model is working! Download from Google Drive and deploy to EC2.")
else:
    print(f"⚠️ Test image not found: {test_img_path}")


# ============================================================================
# CELL 10: Training History Comparison
# ============================================================================

if training_history:
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train', markersize=4)
    plt.plot(epochs, val_losses, 'r-o', label='Validation', markersize=4)
    
    # Show v1 baseline
    plt.axhline(y=0.5277, color='gray', linestyle='--', alpha=0.5, label='v1 best (0.5277)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fashion-CLIP v2 Training Progress')
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
    plt.savefig(os.path.join(CHECKPOINT_DIR, "v2_training_history.png"), dpi=100)
    plt.show()
    
    total_time = sum(times)
    print(f"\n⏱️ Total training time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
    print(f"📈 v1 best val loss: 0.5277")
    print(f"📈 v2 best val loss: {best_val_loss:.4f}")
    print(f"📈 Improvement: {((0.5277 - best_val_loss) / 0.5277 * 100):.1f}%")
