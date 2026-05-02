# Multi-Model Search & Upload Plan

This plan outlines the changes to support multiple CLIP models and an image upload feature in the Streamlit app.

## Proposed Changes

### `app.py`
[MODIFY] `app.py`
We will redesign the app to support dynamic model selection and image management:

#### 1. Model Management (Sidebar)
- **Model Selection**: A dropdown menu with popular CLIP models (e.g., `clip-ViT-L-14`, `clip-ViT-B-32`, `clip-ViT-B-16`).
- **Custom Model**: A text input to allow users to specify and download any other `sentence-transformers` compatible CLIP model.
- **Model Switching**: Selecting a new model will trigger:
    - Loading the model (with `@st.cache_resource`).
    - Switching to a model-specific ChromaDB collection (e.g., `image_search_<model_name>`) to ensure embedding dimension compatibility.

#### 2. Image Upload & Vectorization (Sidebar)
- **Upload Utility**: An `st.file_uploader` for adding new images.
- **Vectorization Button**: A button that:
    1. Saves images to `images/`.
    2. Generates embeddings using the **currently selected model**.
    3. Adds them to the model's specific collection.

#### 3. Search Interface (Main)
- The search functionality will automatically use the active model and its corresponding database collection.

## Verification Plan

### Manual Verification
- **Model Switching**: Switch between two different models and verify that search results change (as they are using different embeddings/collections).
- **Custom Model**: Enter a new model name (e.g., `clip-ViT-B-16`), wait for it to download, and verify it can be used for search.
- **Dynamic Ingestion**: Upload an image and vectorize it for a specific model, then verify it's searchable only when that model is active.
