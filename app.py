import os
import streamlit as st
from PIL import Image
import chromadb

# Redirect all caches to E: drive BEFORE importing ML libraries
HF_CACHE = os.path.abspath("hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

from sentence_transformers import SentenceTransformer

# Configuration
DB_PATH = "chroma_db"
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

st.set_page_config(page_title="Image Vector Search", layout="wide", page_icon="🖼️")

@st.cache_resource
def get_chromadb_client():
    return chromadb.PersistentClient(path=DB_PATH)

@st.cache_resource
def get_model(model_name):
    return SentenceTransformer(model_name)

def get_collection(client, model_name):
    # Dynamic collection name based on model to avoid dimension mismatch
    collection_name = f"image_search_{model_name.replace('-', '_').replace('.', '_')}_v2"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def main():
    st.title("🖼️ Image Vector Search")
    
    client = get_chromadb_client()
    
    # --- Sidebar: Model Management ---
    st.sidebar.title("Settings")
    model_options = ["clip-ViT-B-32"]
    selected_model_name = st.sidebar.selectbox("Select CLIP Model", model_options + ["Custom..."])
    
    if selected_model_name == "Custom...":
        selected_model_name = st.sidebar.text_input("Enter model name (sentence-transformers)", "clip-ViT-B-16")
    
    with st.sidebar:
        st.info(f"Active Model: `{selected_model_name}`")
        try:
            with st.spinner("Loading model..."):
                model = get_model(selected_model_name)
                collection = get_collection(client, selected_model_name)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

    # --- Sidebar: Image Upload & Vectorization ---
    st.sidebar.divider()
    st.sidebar.subheader("Upload & Vectorize")
    uploaded_files = st.sidebar.file_uploader("Choose images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files and st.sidebar.button("Process & Vectorize"):
        with st.sidebar:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                filepath = os.path.join(IMAGE_DIR, filename)
                
                try:
                    status_text.text(f"Processing {filename}...")
                    
                    # Open the image, convert to RGB, and resize to 224x224
                    image = Image.open(uploaded_file)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = image.resize((224, 224))
                    
                    # Save the optimized image
                    image.save(filepath)
                    
                    # Generate embedding
                    embedding = model.encode(image).tolist()
                    
                    collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        metadatas=[{"path": filepath}]
                    )
                    st.success(f"Added {filename}")
                except Exception as e:
                    st.error(f"Error {filename}: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")

    # --- Main: Search Interface ---
    st.markdown("Search for images using natural language, powered by **CLIP** and **ChromaDB**.")
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter search query:", placeholder="e.g., 'a serene landscape'")
    with col2:
        n_results = st.slider("Number of results", min_value=1, max_value=20, value=6)
    
    if query:
        with st.spinner("Searching..."):
            try:
                query_embedding = model.encode(query).tolist()
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                if results['ids'] and results['ids'][0]:
                    st.subheader(f"Results for '{query}'")
                    cols = st.columns(3)
                    for i in range(len(results['ids'][0])):
                        col_idx = i % 3
                        image_id = results['ids'][0][i]
                        distance = results['distances'][0][i]
                        metadata = results['metadatas'][0][i]
                        image_path = metadata['path']
                        
                        with cols[col_idx]:
                            st.container(border=True)
                            if os.path.exists(image_path):
                                st.image(image_path, width='stretch')
                                st.caption(f"**File:** {image_id}  \n**Distance:** `{distance:.4f}`")
                            else:
                                st.warning(f"File not found: {image_path}")
                else:
                    st.info("No matching images found.")
            except Exception as e:
                st.error(f"Search error: {e}")

if __name__ == "__main__":
    main()
