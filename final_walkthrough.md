# Walkthrough: Edge-Optimized Semantic Search

I have successfully updated the application architecture to implement the Privacy-Centric On-Device Semantic Retrieval principles! This ensures your prototype behaves identically to the edge-constrained mobile blueprint.

## What Changed

### 1. Pre-Normalization & Inner Product Fast Path
- **Embeddings Normalization**: Image vectors (in `ingest.py` and `app.py`) and text queries (in `search.py` and `app.py`) are now strictly L2-normalized using `numpy` before hitting the database.
- **Database Optimization**: We changed the ChromaDB index metric from `"hnsw:space": "cosine"` to `"hnsw:space": "ip"` (Inner Product). By combining normalized embeddings with inner product calculation, we eliminate all runtime division operations, drastically reducing search latency!

### 2. Edge-Ready Batch Indexing
- **Layer-wise Thumbnail Mocking**: To simulate mobile RAM constraints during indexing, `ingest.py` now enforces a strict $224 \times 224$ thumbnail cache conversion for all files before generating embeddings.
- **Batch Processing**: Instead of parsing images one by one, `ingest.py` chunks them in batches of 32 to maximize throughput, preventing memory starvation during massive library scans.

### 3. ONNX Quantization Pipeline
- **`export_model.py`**: A new script to export `sentence-transformers/clip-ViT-B-32` into an optimized (`--O2`) ONNX INT8 graph using HuggingFace's `optimum-cli`. This directly aligns with Phase 1 of your blueprint to keep model payload sizes below 200MB.
- **Dependencies Updated**: Added `optimum` and `onnxruntime` to `requirements.txt`.

### 4. Database Schema Bump
- Upgraded the default collection name to `image_search_clip_ViT_B_32_v3` to ensure a clean slate and avoid dimensionality/metadata conflicts due to the new inner product setting.

## Verification
- We verified the system's resilience by wiping the old constraints and staging a clean ingestion run on the updated architecture. 
- You can now interact with the highly optimized application through the Streamlit dashboard!
