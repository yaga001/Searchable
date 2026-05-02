import os

# Redirect all caches to E: drive BEFORE importing ML libraries
HF_CACHE = os.path.abspath("hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

from sentence_transformers import SentenceTransformer
from PIL import Image
import chromadb
import argparse
import numpy as np

# Configuration
IMAGE_DIR = "images"
DB_PATH = "chroma_db"

def main():
    parser = argparse.ArgumentParser(description="Ingest images into ChromaDB using a specified CLIP model.")
    parser.add_argument("--model", type=str, default="clip-ViT-B-32", help="The sentence-transformers CLIP model to use.")
    args = parser.parse_args()

    # Dynamic collection name based on model to avoid dimension mismatch
    collection_name = f"image_search_{args.model.replace('-', '_').replace('.', '_')}_v3"

    # 1. Initialize ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Initialize the Model (CLIP)
    # We use sentence-transformers CLIP implementation
    model = SentenceTransformer(args.model)
    
    # 3. Create or get collection
    # Note: Chroma handles storage of IDs, embeddings, and metadata
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"}
    )
    
    # 4. Process images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}. Please add some images first.")
        return

    print(f"Found {len(image_files)} images. Starting ingestion...")
    
    batch_size = 32
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        images = []
        valid_filenames = []
        valid_filepaths = []
        
        for filename in batch_files:
            filepath = os.path.join(IMAGE_DIR, filename)
            try:
                # Load, convert to RGB, and resize to 224x224 thumbnail
                image = Image.open(filepath)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = image.resize((224, 224))
                images.append(image)
                valid_filenames.append(filename)
                valid_filepaths.append(filepath)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        if not images:
            continue

        try:
            print(f"Embedding batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}...")
            # Generate embeddings
            embeddings = model.encode(images)
            
            # Pre-normalize for dot product (ip) space
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = (embeddings / norms).tolist()
            
            # Add to ChromaDB
            collection.add(
                ids=valid_filenames,
                embeddings=normalized_embeddings,
                metadatas=[{"path": path} for path in valid_filepaths]
            )
            print(f"Successfully ingested {len(valid_filenames)} images")
        except Exception as e:
            print(f"Error embedding batch: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
