import os
import argparse
import numpy as np
from PIL import Image
import chromadb
import cv2
from tqdm import tqdm
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
    collection_name = f"visionai_v4_{args.model.replace('-', '_').replace('.', '_')}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"}
    )

    def process_video_cli(filepath, filename, model, collection):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"Error: Could not open video {filename}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Processing video {filename} ({int(duration)}s)...")
        for second in tqdm(range(int(duration))):
            frame_idx = int(second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            clip_img = pil_img.resize((224, 224))
            
            raw_emb = model.encode(clip_img, show_progress_bar=False)
            embedding = (raw_emb / np.linalg.norm(raw_emb)).tolist()
            
            collection.add(
                ids=[f"{filename}_s{second}"],
                embeddings=[embedding],
                metadatas=[{"path": filepath, "timestamp": float(second), "type": "video", "name": filename}]
            )
        cap.release()

    # 3. Process Files
    all_files = [f for f in os.listdir(args.image_dir)]
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    video_files = [f for f in all_files if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    if not image_files and not video_files:
        print(f"No media found in {args.image_dir}.")
        return

    # Process Videos First
    for filename in video_files:
        process_video_cli(os.path.join(args.image_dir, filename), filename, model, collection)

    # Process Images
    if image_files:
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
                    
                    # Resize only for CLIP
                    clip_img = image.resize((224, 224))
                    images.append(clip_img)
                    valid_filenames.append(filename)
                    valid_filepaths.append(filepath)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

            if images:
                embeddings = model.encode(images)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized_embeddings = (embeddings / norms).tolist()
                
                collection.add(
                    ids=valid_filenames,
                    embeddings=normalized_embeddings,
                    metadatas=[{"path": path, "type": "image"} for path in valid_filepaths]
                )
                print(f"Batch {i//batch_size + 1}: Added {len(valid_filenames)} images.")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
