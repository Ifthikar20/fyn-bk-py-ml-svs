# Fashion-CLIP Training & Deployment Guide

## What You'll Build
Fine-tune CLIP on 44K fashion images to replace EfficientNet's generic captioning. The model will understand "purple striped men's jacket" instead of returning "makeup."

---

## Step 1: Prepare Kaggle Access (5 min)

1. Go to [kaggle.com](https://kaggle.com) → Sign up (or log in)
2. Go to **Settings** → **API** → **Create New Token**
3. Download `kaggle.json` — you'll upload this to Colab

## Step 2: Open the Training Notebook (2 min)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime** → **Change runtime type** → Select **T4 GPU**
3. Create a new notebook
4. Open `FYNDA_ML_Services/notebooks/train_fashion_clip.py` (in your project)
5. Copy each `# CELL X` section into a separate Colab cell

### Cell Order
| Cell | What It Does | Time |
|------|-------------|------|
| 1 | Install dependencies | ~2 min |
| 2 | Download fashion dataset from Kaggle | ~15 min |
| 3 | Explore dataset stats | ~1 min |
| 4 | Create text descriptions from metadata | ~3 min |
| 5 | Build PyTorch dataset + load CLIP | ~3 min |
| 6 | Configure training (freeze layers, optimizer) | ~1 min |
| 7 | **Train!** (the long one) | **4-8 hours** |
| 8 | Export model to Google Drive | ~1 min |
| 9 | Test the fine-tuned model | ~2 min |
| 10 | Plot training history | ~1 min |

## Step 3: Download the Model (5 min)

After training completes:
1. Go to **Google Drive** → `fynda_fashion_clip/export/`
2. Download two files:
   - `fashion_clip_fynda.pt` (~350MB)
   - `config.json`

## Step 4: Deploy to EC2 (10 min)

```bash
# 1. Create the model directory on your Mac
mkdir -p FYNDA_ML_Services/models/fashion_clip/

# 2. Move the downloaded files there
mv ~/Downloads/fashion_clip_fynda.pt FYNDA_ML_Services/models/fashion_clip/
mv ~/Downloads/config.json FYNDA_ML_Services/models/fashion_clip/

# 3. Rsync to EC2
rsync -avz --progress -e "ssh -i ~/.ssh/fynda-api-key.pem" \
  FYNDA_ML_Services/ ubuntu@54.81.148.134:/home/ubuntu/fynda/FYNDA_ML_Services/

# 4. SSH into EC2 and rebuild
ssh -i ~/.ssh/fynda-api-key.pem ubuntu@54.81.148.134
cd /home/ubuntu/fynda
docker compose -f docker-compose.prod.yml build ml
docker compose -f docker-compose.prod.yml up -d ml

# 5. Verify — check logs for "Loaded fine-tuned Fashion-CLIP"
docker compose -f docker-compose.prod.yml logs ml --tail=20
```

## Step 5: Test Image Search

Upload an image on [fynda.shop](https://fynda.shop) — the ML service will now use Fashion-CLIP instead of EfficientNet for describing uploaded images.

---

## Files Created

| File | Purpose |
|------|---------|
| `notebooks/train_fashion_clip.py` | Colab training script (copy cells into notebook) |
| `app/models/fashion_clip.py` | ML service integration (auto-loads the model) |
| `models/fashion_clip/` | Directory for the trained model (create after training) |

## RAM Consideration

The fine-tuned model uses ~800MB RAM. Your `t2.medium` has 4GB total. If you see OOM errors, upgrade to `t2.large` (8GB, ~$0.09/hr).
