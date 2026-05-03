import os
import argparse
import numpy as np
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration & Cache Redirection ---
HF_CACHE = os.path.abspath("hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

def main():
    parser = argparse.ArgumentParser(description="Ingest images into ChromaDB using a specified CLIP model.")
    parser.add_argument("--model", type=str, default="clip-ViT-B-32", help="The sentence-transformers CLIP model to use.")
    parser.add_argument("--offline", action="store_true", help="Force offline mode (do not check for updates).")
    parser.add_argument("--image_dir", type=str, default="images", help="Directory containing images.")
    parser.add_argument("--db_path", type=str, default="chroma_db", help="Path to ChromaDB storage.")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("🔒 Offline mode active. Using cached model files only.")

    # 1. Initialize Model
    print(f"Loading model: {args.model}...")
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        if args.offline:
            print("Hint: The model might not be in the cache. Try running without --offline once.")
        return

    # 2. Initialize ChromaDB
    client = chromadb.PersistentClient(path=args.db_path)
    collection_name = f"image_search_{args.model.replace('-', '_').replace('.', '_')}_v3"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"}
    )

    # 3. Process Images
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not image_files:
        print(f"No images found in {args.image_dir}.")
        return

    print(f"Found {len(image_files)} images. Starting ingestion...")
    
    batch_size = 32
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        images = []
        valid_filenames = []
        valid_filepaths = []
        
        for filename in batch_files:
            filepath = os.path.join(args.image_dir, filename)
            try:
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
            print(f"Embedding batch {i//batch_size + 1}...")
            embeddings = model.encode(images)
            
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = (embeddings / norms).tolist()
            
            collection.add(
                ids=valid_filenames,
                embeddings=normalized_embeddings,
                metadatas=[{"path": path} for path in valid_filepaths]
            )
            print(f"Added {len(valid_filenames)} images.")
        except Exception as e:
            print(f"Error embedding batch: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
