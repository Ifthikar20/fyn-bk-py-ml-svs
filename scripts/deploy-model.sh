#!/bin/bash
# Deploy a new Fashion CLIP model to EC2
# Usage: ./deploy-model.sh <path-to-model.pt>
#
# Example:
#   ./deploy-model.sh ~/Downloads/fashion_clip_v4_best.pt
#
set -e

# --- Configuration ---
EC2_HOST="54.81.148.134"
EC2_USER="ubuntu"
SSH_KEY="${FYNDA_SSH_KEY:-$HOME/downloads/fynda_downloads/fynda-deploy.pem}"
PROJECT_DIR="/home/ubuntu/fynda"
MODEL_DIR="$PROJECT_DIR/FYNDA_ML_Services/models/fashion_clip"

# --- Validate ---
MODEL_PATH="${1:?Usage: ./deploy-model.sh <path-to-model.pt>}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ File not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
    echo "❌ SSH key not found: $SSH_KEY"
    echo "   Set FYNDA_SSH_KEY env var or place key at default path"
    exit 1
fi

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no"
MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)

echo "🚀 Deploying Fashion CLIP model to EC2"
echo "   Model: $MODEL_PATH ($MODEL_SIZE)"
echo "   Target: $EC2_USER@$EC2_HOST:$MODEL_DIR/"
echo ""

# Step 1: Upload model
echo "📤 Uploading model..."
scp $SSH_OPTS "$MODEL_PATH" "$EC2_USER@$EC2_HOST:$MODEL_DIR/fashion_clip_fynda.pt"
echo "   ✅ Model uploaded"

# Step 2: Also upload config.json if it exists next to the model
CONFIG_DIR="$(dirname "$MODEL_PATH")"
if [ -f "$CONFIG_DIR/config.json" ]; then
    echo "📤 Uploading config.json..."
    scp $SSH_OPTS "$CONFIG_DIR/config.json" "$EC2_USER@$EC2_HOST:$MODEL_DIR/config.json"
    echo "   ✅ Config uploaded"
fi

# Step 3: Restart ML container
echo "🔄 Restarting ML container..."
ssh $SSH_OPTS "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && docker compose -f docker-compose.prod.yml restart ml"
echo "   ✅ ML container restarted"

# Step 4: Wait for model to load and health check
echo "⏳ Waiting for model to load (30s)..."
sleep 30

echo "🏥 Health check..."
ssh $SSH_OPTS "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && docker compose -f docker-compose.prod.yml logs ml --tail=5"

# Step 5: Clear search cache
echo "🧹 Clearing search cache..."
ssh $SSH_OPTS "$EC2_USER@$EC2_HOST" "cd $PROJECT_DIR && docker compose -f docker-compose.prod.yml exec -T api python manage.py shell -c 'from django.core.cache import cache; cache.clear(); print(\"Cache cleared\")'" 2>/dev/null

echo ""
echo "✅ Model deployment complete!"
echo "   Test: curl 'https://api.fynda.shop/api/search/?q=blue+tote+bag&limit=3'"
