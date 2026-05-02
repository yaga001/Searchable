# Image Vector Search - Project Walkthrough

This document outlines the successful implementation of the image vector search application. The system enables natural language searching over a collection of local images by leveraging CLIP embeddings and ChromaDB.

## Architecture

The project consists of a straightforward two-script architecture:
- **`ingest.py`**: Processes images, generates embeddings using `clip-ViT-B-32`, and stores them alongside their metadata in ChromaDB.
- **`search.py`**: Takes a natural language query, encodes it using the same CLIP model, and queries ChromaDB for the closest image matches in vector space.

## What Was Done

1. **Environment Initialization**: 
    - Created a 64-bit Python 3.12 virtual environment to support necessary ML packages.
    - Configured caching and temporary file paths to utilize the `E:` drive due to space constraints on the primary drive.
    - Installed required libraries: `sentence-transformers`, `chromadb`, `Pillow`, and `torch`.
2. **Implementation**:
    - Wrote the ingestion script to iterate through the `images/` directory and populate the vector database.
    - Wrote the search script to provide a command-line interface for querying the database and returning the top 3 visually relevant images.
3. **Execution**:
    - Added 7 sample images.
    - Successfully embedded the images and populated the local ChromaDB database.
    - Ran tests.

## Validation Results

The application performed as expected during testing.

For example, when querying for "a black dog":
```
Searching for: 'a black dog'...

Search Results:
1. a Dog Doberman.jpg (Distance: 152.0568)
   Path: images\a Dog Doberman.jpg
2. A girl Design poster • pretty eyes • pretty heart •….jpg (Distance: 155.3349)
   Path: images\A girl Design poster • pretty eyes • pretty heart •….jpg
3. man smoking .jpeg (Distance: 158.8405)
   Path: images\man smoking .jpeg
```

When querying for "a assualt rifle":
```
Searching for: 'a assualt rifle'...

Search Results:
1. A gun d8tkqjypt96b1-removebg-preview.png (Distance: 147.0354)
   Path: images\A gun d8tkqjypt96b1-removebg-preview.png
2. A girl Design poster • pretty eyes • pretty heart •….jpg (Distance: 159.1917)
   Path: images\A girl Design poster • pretty eyes • pretty heart •….jpg
3. Christoph Waltz by Alasdair McLellan.jpeg (Distance: 165.4159)
   Path: images\Christoph Waltz by Alasdair McLellan.jpeg
```

> [!NOTE]
> The search results successfully return the most semantically relevant images as the top matches based on their L2 distance.

## Future Possibilities

If you want to continue extending this application, some potential next steps could include:
- Adding a simple web UI (e.g., using Streamlit, Flask, or a React frontend) to visually display the search results instead of just logging paths.
- Implementing support for updating or deleting specific images in the database without re-ingesting everything.
- Experimenting with larger or fine-tuned CLIP models for better accuracy.
