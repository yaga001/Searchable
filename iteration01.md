# Iteration 01 - Project Setup & Foundation

In this iteration, I have established the core environment and basic application logic for the image vector search system.

## Intentions
- [x] Create a dedicated folder structure for the project.
- [x] Set up a Python virtual environment.
- [x] Install necessary libraries: `sentence-transformers`, `chromadb`, `Pillow`, and `torch`.
- [x] Implement the image ingestion logic (`ingest.py`).
- [x] Implement the text search logic (`search.py`).

## Accomplishments
- **Environment Setup**: 
    - Created the `images/` directory for input files.
    - Successfully configured a 64-bit Python 3.12 environment after identifying a 32-bit limitation.
    - Overcame `C:` drive disk space limitations by redirecting `pip` cache and temporary files to the `E:` drive.
- **Dependency Management**:
    - Installed all ML and database requirements in the `venv`.
- **Code Implementation**:
    - **`ingest.py`**: A script that reads images from the `images` folder, converts them to embeddings using the `clip-ViT-B-32` model, and stores them in a local ChromaDB instance.
    - **`search.py`**: A command-line utility that takes a text query, converts it to an embedding, and retrieves the most visually similar images from ChromaDB.

## Verification
- Added 7 sample images to the `images/` folder.
- Successfully ran `python ingest.py` to populate the ChromaDB database with CLIP embeddings.
- Successfully tested `python search.py "a black dog"` and `python search.py "a assualt rifle"`, receiving accurate semantic visual search results based on textual queries.

## Optimization & Bug Fixes
- **Disk Space Optimization:** We were running low on disk space due to large model caches (`clip-ViT-L-14` and `clip-ViT-B-16`). We removed these unnecessary models and standardized entirely on `clip-ViT-B-32`. 
- **Image Resizing Optimization:** We implemented logic in `app.py` to automatically convert uploaded images to RGB and resize them to 224x224 (CLIP's native resolution) before saving them to disk. This drastically reduces storage requirements and speeds up file loading without affecting the quality of the embeddings.
- **Search Accuracy Bug:** We encountered an issue where search results were highly inaccurate, returning unrelated images with large distance scores (160+).
    - **Root Cause:** By default, ChromaDB uses the L2 (Squared Euclidean) distance metric. However, the `sentence-transformers` CLIP implementation returns unnormalized vectors. L2 distance on unnormalized vectors is heavily distorted by vector magnitude rather than semantic angle.
    - **Solution:** We modified the collection creation logic in `app.py`, `ingest.py`, and `search.py` to explicitly use **Cosine Similarity** (`metadata={"hnsw:space": "cosine"}`). We also versioned the collection name to `_v2` to ensure a clean slate, resulting in highly accurate semantic search capabilities.
