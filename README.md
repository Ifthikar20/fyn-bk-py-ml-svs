# FYNDA ML Services

Visual similarity search API for Fynda marketplace.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Health check
curl http://localhost:8001/api/health
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/visual-search` | Search by image |
| POST | `/api/index-product` | Add product to index |
| GET | `/api/health` | Health check |

## Environment Variables

```
ML_SERVICE_PORT=8001
FAISS_INDEX_PATH=./data/faiss_index
MAX_RESULTS=20
```
