# Visual Search System Explained

## Why "No Similar Products Found"?

The ML service logs show: `Vector store loaded. Total products: 0`

**The search works, but there's nothing to search against.** Products must be indexed first.

---

## How Visual Search Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Upload Image   │────▶│  EfficientNet   │────▶│  1280-dim       │
│  (JPEG/PNG)     │     │  Deep Learning  │     │  Embedding      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Similar Items  │◀────│  FAISS Index    │◀────│  Similarity     │
│  (Results)      │     │  (Product DB)   │     │  Matching       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Step-by-Step:
1. **User uploads image** → Frontend converts to base64
2. **ML Service receives** → `POST /api/visual-search`
3. **EfficientNet encodes** → Creates 1280-dimensional vector
4. **FAISS searches** → Finds nearest neighbors in vector space
5. **Returns matches** → Product IDs + similarity scores

---

## The Problem: Empty Index

The FAISS index is empty. No products have been indexed yet.

To find similar products, we first need to **index products** by:
1. Taking each product image
2. Encoding it with EfficientNet
3. Storing the vector in FAISS

---

## Solution: Index Products

### Option 1: Index from Django Products
```bash
cd FYNDA_ML_Services
source venv/bin/activate
python scripts/index_products.py
```

### Option 2: Manual Index via API
```bash
curl -X POST http://localhost:8001/api/index-product \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "abc123",
    "image_url": "https://example.com/product.jpg",
    "metadata": {"title": "Blue Dress", "price": 49.99}
  }'
```

---

## Current Status

| Component | Status |
|-----------|--------|
| ML Service | ✅ Running on :8001 |
| EfficientNet | ✅ Loaded (1280 dim) |
| FAISS Index | ⚠️ Empty (0 products) |
| Visual Search | ✅ Working (no data) |

---

## Next Step

**Run the indexing script** to populate the FAISS index with product embeddings from your Django database.
