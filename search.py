import os

# Redirect all caches to E: drive BEFORE importing ML libraries
HF_CACHE = os.path.abspath("hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

import sys
import argparse
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration
DB_PATH = "chroma_db"

def search(query_text, model_name, n_results=3):
    # Dynamic collection name based on model to avoid dimension mismatch
    collection_name = f"image_search_{model_name.replace('-', '_').replace('.', '_')}_v2"

    # 1. Initialize ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=collection_name)
    
    # 2. Initialize the Model (CLIP)
    model = SentenceTransformer(model_name)
    
    # 3. Embed the text query
    print(f"Searching for: '{query_text}'...")
    query_embedding = model.encode(query_text).tolist()
    
    # 4. Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # 5. Display results
    print("\nSearch Results:")
    for i in range(len(results['ids'][0])):
        image_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        print(f"{i+1}. {image_id} (Distance: {distance:.4f})")
        print(f"   Path: {metadata['path']}")

def main():
    parser = argparse.ArgumentParser(description="Search images using a specified CLIP model.")
    parser.add_argument("query", nargs="*", help="The text query to search for.")
    parser.add_argument("--model", type=str, default="clip-ViT-B-32", help="The sentence-transformers CLIP model to use.")
    args = parser.parse_args()

    if args.query:
        query = " ".join(args.query)
    else:
        query = input("Enter search query: ")
    
    try:
        search(query, args.model)
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    main()
