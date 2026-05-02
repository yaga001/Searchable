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

# Configuration
IMAGE_DIR = "images"
DB_PATH = "chroma_db"

def main():
    parser = argparse.ArgumentParser(description="Ingest images into ChromaDB using a specified CLIP model.")
    parser.add_argument("--model", type=str, default="clip-ViT-B-32", help="The sentence-transformers CLIP model to use.")
    args = parser.parse_args()

    # Dynamic collection name based on model to avoid dimension mismatch
    collection_name = f"image_search_{args.model.replace('-', '_').replace('.', '_')}_v2"

    # 1. Initialize ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # 2. Initialize the Model (CLIP)
    # We use sentence-transformers CLIP implementation
    model = SentenceTransformer(args.model)
    
    # 3. Create or get collection
    # Note: Chroma handles storage of IDs, embeddings, and metadata
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 4. Process images
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}. Please add some images first.")
        return

    print(f"Found {len(image_files)} images. Starting ingestion...")
    
    for filename in image_files:
        filepath = os.path.join(IMAGE_DIR, filename)
        try:
            print(f"Processing {filename}...")
            
            # Load and embed image
            image = Image.open(filepath)
            embedding = model.encode(image).tolist()
            
            # Add to ChromaDB
            collection.add(
                ids=[filename],
                embeddings=[embedding],
                metadatas=[{"path": filepath}]
            )
            print(f"Successfully ingested {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
