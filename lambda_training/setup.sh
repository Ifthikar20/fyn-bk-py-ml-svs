#!/bin/bash
# =============================================================================
# Lambda Cloud Setup — Fashion CLIP v4 Training
# Run this ONCE after SSH-ing into the instance
# =============================================================================
set -e

echo "🚀 Setting up Fashion CLIP v4 training environment..."

# Install Python dependencies
pip install -q transformers torch torchvision pillow kagglehub tqdm open_clip_torch datasets

# Create directories
mkdir -p ~/data/fashionpedia_images
mkdir -p ~/checkpoints
mkdir -p ~/export

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Upload your v3 checkpoint (from your Mac):"
echo "     scp FYNDA_ML_Services/models/fashion_clip/fashion_clip_fynda.pt ubuntu@<IP>:~/checkpoints/best_model_v3.pt"
echo ""
echo "  2. Run training:"
echo "     python ~/train_v4_lambda.py"
echo ""
