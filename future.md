# Future Roadmap: Multimodal Edge Semantic Search

This document outlines the strategic evolution of the project from a photo-search utility to a comprehensive, cross-platform multimedia and document semantic retrieval engine.

## 📱 1. Deep Android Optimizations

To move from Python/Desktop to a native Android production environment, the following hardware-algorithm co-design strategies should be implemented:

- **Hardware Delegates**: Implement **NNAPI (Neural Network API)** for real-time query encoding on NPUs and **GPU Delegates** for high-throughput background indexing.
- **Speculative Retrieval**: Implement query embeddings at multiple granularities. Use a "coarse-to-fine" strategy where early exits from the model filter the top-K candidates, followed by refinement.
- **Progressive LoRA (P-LoRA)**: Instead of re-computing embeddings, use shared weights across layers to allow activation reuse, achieving up to 14x throughput improvements on mobile chipsets.
- **Scalar Quantization (SQ-8)**: Store vectors as INT8 instead of FP32. This reduces the storage footprint of a 100k+ library by ~75% while minimizing I/O latency.

## 🎬 2. Expanding to Video Search

Semantic video retrieval requires temporal understanding beyond static frames:

- **Keyframe Extraction**: Implement a scene-change detection algorithm (using Histograms or Lightness changes) to index only significant "moment" frames.
- **Temporal Aggregation**: Use a **Temporal Shift Module (TSM)** or simple mean-pooling of frame embeddings within a 2-5 second window to represent "actions" (e.g., searching for "blowing out birthday candles").
- **Streaming Indexing**: Process video files in the background using `WorkManager`, utilizing low-res streams to prevent thermal throttling.

## 📄 3. Semantic Document Retrieval (OCR + Embeddings)

Transforming the engine into a file search solution:

- **Hybrid Search**: Combine traditional full-text search (BM25) with semantic embeddings.
- **Visual OCR**: For PDFs and screenshots, use a lightweight OCR engine (like Tesseract or ML Kit) to extract text, then embed that text using a multilingual model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`).
- **Document CLIP**: Use models trained on document-image pairs to search through diagrams, charts, and scanned invoices semantically.

## 🧠 4. Advanced "Recall" Framework

To achieve high accuracy with minimal energy:

- **Speculative Caching**: Cache query results for common semantic concepts locally.
- **Lazy Vector Indexes**: Only update the vector index during idle charging states to prevent degrading user-perceived device performance.
- **Multimodal Fusion**: Map text, images, and audio into the *same* latent space using models like **ImageBind**, allowing a query to find a photo of a cat, a video of a cat, or the sound of a cat meowing simultaneously.

---
*Target: 100% Offline, Zero-Latency, Universal Semantic Discovery.*
