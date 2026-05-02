# Edge-Optimized Semantic Search

This project implements a privacy-centric, on-device semantic image search system using CLIP (Contrastive Language-Image Pre-training) and ChromaDB. It is architected to mirror the constraints and optimizations required for mobile/edge deployment.

## 🚀 Key Features

- **Multimodal Search**: Query your local image library using natural language.
- **100% Offline & Private**: All embeddings and vector searches are performed locally on your hardware.
- **Edge-Optimized Architecture**: 
  - **Dot Product Fast-Path**: Uses L2-normalized embeddings for lightning-fast inner product similarity.
  - **Memory-Aware Indexing**: Automatically generates 224x224 thumbnails to stay within mobile memory limits.
  - **ONNX Pipeline Ready**: Includes scripts to export and quantize models to INT8 (< 200MB) for production edge runtimes.

## 🛠️ Project Structure

- `app.py`: The main Streamlit dashboard for searching and uploading images.
- `ingest.py`: Optimized bulk ingestion script for processing large image folders.
- `search.py`: Command-line interface for semantic queries.
- `export_model.py`: Utility to export CLIP models to optimized ONNX format.
- `images/`: Local directory for indexed image assets.
- `chroma_db/`: Persistent local vector store.

## 🚦 Getting Started

### 1. Installation
Ensure you have Python 3.10+ and a virtual environment active:
```powershell
pip install -r requirements.txt
```

### 2. Ingest Images
Place your images in the `images/` folder and run the optimized ingestion pipeline:
```powershell
python ingest.py
```

### 3. Launch the Dashboard
Start the Streamlit interface to browse and search:
```powershell
streamlit run app.py
```

### 4. Export for Edge (Optional)
To generate a quantized ONNX model for mobile deployment:
```powershell
python export_model.py
```

## 🏗️ Technical Specifications

- **Model**: `clip-ViT-B-32` (Sentence-Transformers) / MobileCLIP (ONNX Support).
- **Vector Space**: HNSW with Inner Product (`ip`).
- **Input Resolution**: 224x224 (Normalized).
- **Optimization**: speculative retrieval and speculative caching alignment.

---
*Developed as part of the Privacy-Centric On-Device Semantic Retrieval Blueprint.*
