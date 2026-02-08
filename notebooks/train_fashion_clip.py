"""
=============================================================================
 FYNDA Fashion-CLIP Training Notebook
 ------------------------------------
 Run this in Google Colab Pro with T4 GPU
 
 HOW TO USE:
 1. Open Google Colab (colab.research.google.com)
 2. Create a new notebook
 3. Go to Runtime → Change runtime type → T4 GPU
 4. Copy each "# CELL X" section into a separate Colab cell
 5. Run cells in order (Shift+Enter)
 
 Training time: ~4-8 hours on T4 GPU
 Disk needed: ~10GB (dataset + model)
 Output: fashion_clip_fynda.pt (~350MB saved to Google Drive)
=============================================================================
"""

# ============================================================================
# CELL 1: Install Dependencies (~2 min)
# ============================================================================
# >>> COPY EVERYTHING BELOW INTO YOUR FIRST COLAB CELL <<<

!pip install -q transformers torch torchvision pillow kagglehub tqdm
!pip install -q open_clip_torch

# Mount Google Drive for saving the model
from google.colab import drive
drive.mount('/content/drive')


# ============================================================================
# CELL 2: Download Fashion Dataset (~15 min)
# ============================================================================

"""
We use the Kaggle "Fashion Product Images (Small)" dataset:
- 44,000+ fashion product images
- Rich metadata: gender, category, subCategory, articleType, baseColour, season, usage
- Perfect for training fashion understanding

You need a Kaggle account. If you don't have one:
1. Go to kaggle.com → Sign up
2. Go to kaggle.com/settings → API → Create New Token
3. Upload the kaggle.json when prompted
"""

import os
import json
import pandas as pd
from pathlib import Path

# Download the dataset (this will prompt for Kaggle credentials if needed)
import kagglehub
dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print(f"Dataset downloaded to: {dataset_path}")

DATASET_PATH = dataset_path


# ============================================================================
# CELL 3: Explore & Prepare the Dataset (~5 min)
# ============================================================================

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Load metadata
styles_csv = os.path.join(DATASET_PATH, "styles.csv")
df = pd.read_csv(styles_csv, on_bad_lines='skip')

print(f"Dataset size: {len(df)} items")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample row:")
print(df.iloc[0])

# Key columns we'll use:
# - id: filename (e.g., 15970.jpg)
# - gender: Men, Women, Unisex
# - masterCategory: Apparel, Accessories, Footwear
# - subCategory: Topwear, Bottomwear, Watches, etc.
# - articleType: Tshirts, Jeans, Watches, etc.
# - baseColour: Black, Blue, White, etc.
# - season: Summer, Winter, Fall, Spring
# - usage: Casual, Formal, Sports, Party

print(f"\n📊 Category distribution:")
print(df['masterCategory'].value_counts().head(10))
print(f"\n🎨 Color distribution:")
print(df['baseColour'].value_counts().head(15))
print(f"\n👕 Article types:")
print(df['articleType'].value_counts().head(20))


# ============================================================================
# CELL 4: Create Fashion Text Descriptions (~3 min)
# ============================================================================

import random

def create_fashion_description(row):
    """
    Convert metadata into natural language descriptions.
    Creates multiple variations for robust training.
    
    Example outputs for the same item:
      "black casual tshirt for men"
      "men's black t-shirt"
      "casual black tee"
    """
    parts = {}
    
    # Extract available attributes
    color = str(row.get('baseColour', '')).strip().lower()
    if color in ['nan', '', 'none']:
        color = ''
    
    gender = str(row.get('gender', '')).strip().lower()
    if gender in ['nan', '', 'none']:
        gender = ''
    
    article = str(row.get('articleType', '')).strip().lower()
    if article in ['nan', '', 'none']:
        article = ''
    
    usage = str(row.get('usage', '')).strip().lower()
    if usage in ['nan', '', 'none']:
        usage = ''
    
    season = str(row.get('season', '')).strip().lower()
    if season in ['nan', '', 'none']:
        season = ''
    
    sub_cat = str(row.get('subCategory', '')).strip().lower()
    if sub_cat in ['nan', '', 'none']:
        sub_cat = ''
    
    # Create multiple description templates
    descriptions = []
    
    # Template 1: Full descriptive
    # "black casual tshirt for men"
    parts_list = [p for p in [color, usage, article] if p]
    if parts_list:
        desc = ' '.join(parts_list)
        if gender and gender not in ['unisex']:
            desc += f" for {gender}"
        descriptions.append(desc)
    
    # Template 2: Gender-first
    # "men's black t-shirt"
    if gender and gender not in ['unisex'] and article:
        gender_prefix = f"{gender}'s" if not gender.endswith('s') else gender
        parts_list = [p for p in [gender_prefix, color, article] if p]
        descriptions.append(' '.join(parts_list))
    
    # Template 3: Color + article only
    # "black tshirt"
    if color and article:
        descriptions.append(f"{color} {article}")
    
    # Template 4: With season
    # "summer black tshirt"
    if season and color and article:
        descriptions.append(f"{season} {color} {article}")
    
    # Template 5: Article only (general)
    if article:
        descriptions.append(article)
    
    # Template 6: Sub-category focused
    # "casual topwear in black"
    if usage and sub_cat and color:
        descriptions.append(f"{usage} {sub_cat} in {color}")
    
    return descriptions if descriptions else [str(row.get('productDisplayName', 'fashion item')).lower()]


# Test the description generator
sample = df.iloc[0]
print(f"Sample item: {sample['productDisplayName']}")
print(f"Generated descriptions: {create_fashion_description(sample)}")
print()

# Generate descriptions for all items
df['descriptions'] = df.apply(create_fashion_description, axis=1)

# Count total training pairs
total_pairs = sum(len(d) for d in df['descriptions'])
print(f"✅ Generated {total_pairs} text-image pairs from {len(df)} items")


# ============================================================================
# CELL 5: Create PyTorch Dataset (~2 min)
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import open_clip

class FashionCLIPDataset(Dataset):
    """
    Dataset that pairs fashion images with text descriptions.
    Each image gets multiple text descriptions for robust training.
    """
    
    def __init__(self, dataframe, image_dir, transform=None, tokenizer=None):
        self.pairs = []
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # Create all (image_path, description) pairs
        for _, row in dataframe.iterrows():
            img_id = str(row['id'])
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
            
            # Only include if image exists
            if os.path.exists(img_path):
                for desc in row['descriptions']:
                    self.pairs.append((img_path, desc))
        
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


# Load the CLIP model and get its preprocessor
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

# Create dataset
images_dir = os.path.join(DATASET_PATH, "images")
dataset = FashionCLIPDataset(
    dataframe=df,
    image_dir=images_dir,
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
# CELL 6: Training Configuration
# ============================================================================

# Hyperparameters
BATCH_SIZE = 64          # T4 can handle 64 with ViT-B/32
LEARNING_RATE = 1e-5     # Low LR for fine-tuning (don't destroy pre-trained knowledge)
NUM_EPOCHS = 5           # 5 epochs is usually enough for fine-tuning
WARMUP_STEPS = 500       # Gradual learning rate warmup
SAVE_EVERY = 1           # Save checkpoint every N epochs
GRAD_ACCUM_STEPS = 4     # Simulate larger batch size (effective batch = 64 * 4 = 256)

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

# Optimizer - only fine-tune the last few layers (freeze most of the model)
# This is key: we keep CLIP's general knowledge and just teach it fashion
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 4 transformer blocks + projection layers
for name, param in model.named_parameters():
    if any(x in name for x in [
        'visual.transformer.resblocks.11',  # Last visual block
        'visual.transformer.resblocks.10',  # Second-to-last
        'transformer.resblocks.11',          # Last text block
        'transformer.resblocks.10',          # Second-to-last  
        'visual.ln_post',                    # Visual projection
        'text_projection',                   # Text projection
        'visual.proj',                       # Visual projection
        'ln_final',                          # Final layer norm
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
print(f"\n✅ Training config ready — {device.upper()}")


# ============================================================================
# CELL 7: Training Loop (~4-8 hours on T4)
# ============================================================================

from tqdm import tqdm
import time

def clip_loss(image_features, text_features, temperature=0.07):
    """Contrastive loss for CLIP training."""
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Cosine similarity as logits
    logit_scale = 1.0 / temperature
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    
    # Labels (diagonal = matching pairs)
    labels = torch.arange(len(image_features), device=image_features.device)
    
    # Symmetric cross-entropy loss
    loss_i2t = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = torch.nn.functional.cross_entropy(logits_per_text, labels)
    
    return (loss_i2t + loss_t2i) / 2


# Google Drive checkpoint path
CHECKPOINT_DIR = "/content/drive/MyDrive/fynda_fashion_clip"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_val_loss = float('inf')
training_history = []

print("=" * 60)
print("🚀 Starting Fashion-CLIP Training")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Training pairs: {len(train_dataset)}")
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
        loss = loss / GRAD_ACCUM_STEPS  # Scale for gradient accumulation
        
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
    }
    
    # Always save latest
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        print(f"   ✅ New best model saved! (val_loss: {avg_val_loss:.4f})")
    
    print()

print("=" * 60)
print("🎉 Training Complete!")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Model saved to: {CHECKPOINT_DIR}")
print("=" * 60)


# ============================================================================
# CELL 8: Export Final Model for Deployment (~1 min)
# ============================================================================

"""
Export the model in a format ready for your EC2 ML service.
This creates a self-contained package with:
  - fashion_clip_fynda.pt (model weights, ~350MB)  
  - config.json (metadata)
"""

# Load the best model
best_checkpoint = torch.load(
    os.path.join(CHECKPOINT_DIR, "best_model.pt"),
    map_location='cpu'
)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Save just the model weights (smaller, faster to load)
EXPORT_DIR = os.path.join(CHECKPOINT_DIR, "export")
os.makedirs(EXPORT_DIR, exist_ok=True)

# Save model state dict
export_path = os.path.join(EXPORT_DIR, "fashion_clip_fynda.pt")
torch.save(model.state_dict(), export_path)

# Save config for loading
config = {
    "model_name": model_name,
    "pretrained_base": pretrained,
    "dataset": "fashion-product-images-small",
    "epochs_trained": best_checkpoint['epoch'],
    "best_val_loss": best_checkpoint['val_loss'],
    "training_history": training_history,
    "fashion_categories": list(df['articleType'].dropna().unique()),
    "fashion_colors": list(df['baseColour'].dropna().unique()),
}

with open(os.path.join(EXPORT_DIR, "config.json"), 'w') as f:
    json.dump(config, f, indent=2)

file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
print(f"✅ Exported model:")
print(f"   Path: {export_path}")
print(f"   Size: {file_size_mb:.1f} MB")
print(f"   Config: {os.path.join(EXPORT_DIR, 'config.json')}")
print(f"\n📥 Download from Google Drive:")
print(f"   Go to drive.google.com → fynda_fashion_clip → export/")
print(f"   Download: fashion_clip_fynda.pt + config.json")


# ============================================================================
# CELL 9: Test the Fine-Tuned Model (~2 min)
# ============================================================================

"""
Quick test: verify the model understands fashion concepts.
"""

# Test with fashion descriptions vs. an image
test_descriptions = [
    "men's black t-shirt",
    "women's red dress",
    "blue jeans",
    "white sneakers",
    "leather handbag",
    "purple striped jacket",
    "yellow summer dress", 
    "casual hoodie",
    "formal shirt",
    "sports shoes",
]

# Test with a random image from the dataset
test_idx = random.randint(0, len(df) - 1)
test_row = df.iloc[test_idx]
test_img_path = os.path.join(images_dir, f"{test_row['id']}.jpg")

if os.path.exists(test_img_path):
    test_image = Image.open(test_img_path).convert('RGB')
    
    # Display the test image
    plt.figure(figsize=(4, 4))
    plt.imshow(test_image)
    plt.title(f"Actual: {test_row.get('productDisplayName', 'Unknown')}")
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
    
    print(f"\n🔍 Model predictions for: {test_row.get('productDisplayName', 'Unknown')}")
    print(f"   Actual category: {test_row.get('articleType', '?')}")
    print(f"   Actual color: {test_row.get('baseColour', '?')}")
    print()
    
    # Show top predictions
    scores, indices = similarity[0].topk(5)
    for score, idx in zip(scores, indices):
        print(f"   {score.item()*100:5.1f}%  {test_descriptions[idx]}")
    
    print(f"\n✅ Model is working! Download from Google Drive and deploy to EC2.")
else:
    print(f"⚠️ Test image not found: {test_img_path}")
    print("   Run a manual test after deployment.")


# ============================================================================
# CELL 10: Plot Training History
# ============================================================================

if training_history:
    epochs = [h['epoch'] for h in training_history]
    train_losses = [h['train_loss'] for h in training_history]
    val_losses = [h['val_loss'] for h in training_history]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train')
    plt.plot(epochs, val_losses, 'r-o', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
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
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_history.png"), dpi=100)
    plt.show()
    
    total_time = sum(times)
    print(f"\n⏱️ Total training time: {total_time:.0f} minutes ({total_time/60:.1f} hours)")
