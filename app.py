import os
import streamlit as st
from PIL import Image
import chromadb
import numpy as np
import time

# --- Configuration & Cache Redirection ---
HF_CACHE = os.path.abspath("hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

# Registry of supported models with metadata
MODEL_INFO = {
    "clip-ViT-B-32": {
        "name": "CLIP ViT-B/32",
        "dims": 512,
        "size": "~605 MB",
        "powerpoints": ["🚀 High Speed", "📉 Low Memory", "✅ General Purpose"],
        "description": "The industry standard for fast, efficient semantic search. Best for large datasets where speed is critical."
    },
    "clip-ViT-B-16": {
        "name": "CLIP ViT-B/16",
        "dims": 512,
        "size": "~605 MB",
        "powerpoints": ["🎯 High Accuracy", "🖼️ Better Detail", "⚖️ Balanced"],
        "description": "Better precision than B/32 by looking at smaller patches. Ideal for detailed scene understanding."
    },
    "clip-ViT-L-14": {
        "name": "CLIP ViT-L/14",
        "dims": 768,
        "size": "~1.7 GB",
        "powerpoints": ["🏆 State-of-the-Art", "🎨 Fine Detail", "🖥️ Heavy Resource"],
        "description": "Maximum accuracy and detail retrieval. Requires significant resources but provides the best results."
    }
}

st.set_page_config(
    page_title="VisionAI | Semantic Search",
    layout="wide",
    page_icon="🖼️",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    .main {
        background: #0e1117;
    }
    .stApp {
        color: #e0e0e0;
    }
    .model-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .power-point {
        display: inline-block;
        background: rgba(0, 255, 127, 0.1);
        color: #00ff7f;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(0, 255, 127, 0.3);
    }
    .stButton>button {
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar: Settings ---
with st.sidebar:
    st.title("⚙️ System Config")
    
    # 1. Connectivity Toggle
    st.subheader("Connectivity")
    online_mode = st.toggle("Online Mode", value=False, help="Enable to download new models or check for updates. Disable for strict offline/cached loading.")
    
    # Set environment variables based on toggle
    if not online_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        st.caption("🔒 **Strict Offline Mode Active**")
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        st.caption("🌐 **Online Mode Active**")

    st.divider()

    # 2. Model Selection
    st.subheader("Model Selection")
    model_options = list(MODEL_INFO.keys())
    selected_model_id = st.selectbox("Select CLIP Architecture", model_options)
    
    # Model Metadata Display (Model Card)
    info = MODEL_INFO[selected_model_id]
    st.markdown(f"""
        <div class="model-card">
            <h4>{info['name']}</h4>
            <p style='font-size: 0.9rem; color: #aaa;'>{info['description']}</p>
            <div>
                {" ".join([f'<span class="power-point">{p}</span>' for p in info['powerpoints']])}
            </div>
            <hr style='margin: 10px 0; border-color: rgba(255,255,255,0.1);'>
            <small>📐 Dims: <b>{info['dims']}</b> | 📦 Size: <b>{info['size']}</b></small>
        </div>
    """, unsafe_allow_html=True)

# Late import to ensure environment variables are set
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_chromadb_client():
    return chromadb.PersistentClient(path="chroma_db")

@st.cache_resource
def load_model(model_name):
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Failed to load model '{model_name}'. If you are offline, ensure the model is already cached.")
        st.info("Try switching to 'Online Mode' to download it.")
        raise e

def get_collection(client, model_name):
    # Dynamic collection name based on model to avoid dimension mismatch
    collection_name = f"image_search_{model_name.replace('-', '_').replace('.', '_')}_v3"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"}
    )

def main():
    st.title("🖼️ VisionAI Semantic Retrieval")
    st.markdown("Search your local image library using natural language embeddings.")

    client = get_chromadb_client()
    
    # Initialize Model & Collection
    try:
        with st.status(f"Initializing {selected_model_id}...", expanded=False) as status:
            model = load_model(selected_model_id)
            collection = get_collection(client, selected_model_id)
            status.update(label="System Ready", state="complete", expanded=False)
    except:
        return

    # --- UI Layout: Search & Upload ---
    tabs = st.tabs(["🔍 Search Library", "📤 Ingest Images"])

    # TABS: SEARCH
    with tabs[0]:
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter search query:", placeholder="e.g., 'a photo of a cat on a beach'")
        with col2:
            n_results = st.slider("Results count", 1, 24, 6)
        
        if query:
            with st.spinner("Analyzing semantics..."):
                try:
                    # Generate query embedding
                    raw_query_emb = model.encode(query)
                    norm = np.linalg.norm(raw_query_emb)
                    query_embedding = (raw_query_emb / norm).tolist()
                    
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results
                    )
                    
                    if results['ids'] and results['ids'][0]:
                        st.subheader(f"Found {len(results['ids'][0])} matches")
                        cols = st.columns(3)
                        for i in range(len(results['ids'][0])):
                            col_idx = i % 3
                            image_id = results['ids'][0][i]
                            distance = results['distances'][0][i]
                            metadata = results['metadatas'][0][i]
                            image_path = metadata['path']
                            
                            with cols[col_idx]:
                                with st.container(border=True):
                                    if os.path.exists(image_path):
                                        st.image(image_path, use_container_width=True)
                                        st.caption(f"**{image_id}** (Score: `{1-distance:.4f}`)")
                                    else:
                                        st.warning(f"File missing: {image_id}")
                    else:
                        st.info("No relevant images found in current collection.")
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # TABS: INGEST
    with tabs[1]:
        st.subheader("Add new images to collection")
        uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Start Vectorization", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            IMAGE_DIR = "images"
            os.makedirs(IMAGE_DIR, exist_ok=True)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                filepath = os.path.join(IMAGE_DIR, filename)
                
                try:
                    status_text.text(f"Processing {filename}...")
                    image = Image.open(uploaded_file)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image = image.resize((224, 224))
                    image.save(filepath)
                    
                    # Embed & Add
                    raw_emb = model.encode(image)
                    embedding = (raw_emb / np.linalg.norm(raw_emb)).tolist()
                    
                    collection.add(
                        ids=[filename],
                        embeddings=[embedding],
                        metadatas=[{"path": filepath}]
                    )
                except Exception as e:
                    st.error(f"Error {filename}: {e}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.success(f"Successfully vectorized {len(uploaded_files)} images!")

if __name__ == "__main__":
    main()
