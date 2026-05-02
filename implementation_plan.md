# Image Vector Search Application

Build a Python-based image search system that uses CLIP embeddings to enable natural language search over a collection of images.

## User Review Required

> [!IMPORTANT]
> The application will download the `clip-ViT-B-32` model (approx. 600MB) from Hugging Face on the first run. Ensure you have a stable internet connection.

## Proposed Changes

### Setup

#### [NEW] [requirements.txt](file:///e:/ANTIGRAVITY/requirements.txt)
Define dependencies: `sentence-transformers`, `chromadb`, `Pillow`.

#### [NEW] [images/](file:///e:/ANTIGRAVITY/images/)
Directory for storing user images.

### Application Logic

#### [NEW] [ingest.py](file:///e:/ANTIGRAVITY/ingest.py)
- Load images from the `images` folder.
- Use `sentence-transformers` (CLIP) to generate embeddings.
- Initialize/Update a local ChromaDB collection.
- Store embeddings and file paths.

#### [NEW] [search.py](file:///e:/ANTIGRAVITY/search.py)
- Take a text query as input.
- Convert text to embedding using the same CLIP model.
- Query ChromaDB for the most similar images.
- Display or list the paths of the matching images.

## Verification Plan

### Automated Tests
- I will run `ingest.py` with a sample image (if available or generated) to ensure the pipeline works.
- I will run `search.py` with a dummy query to verify database retrieval.

### Manual Verification
- User can place images in the `images` folder and run `python ingest.py`.
- User can run `python search.py "a cat"` to find relevant images.
