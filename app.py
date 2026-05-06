import os
import streamlit as st
from PIL import Image
import chromadb
import numpy as np
import time
import cv2
import shutil
from huggingface_hub import snapshot_download

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
        "repo": "sentence-transformers/clip-ViT-B-32",
        "dims": 512,
        "size_mb": 605,
        "powerpoints": ["🚀 High Speed", "📉 Low Memory", "✅ General Purpose"],
        "description": "The industry standard for fast, efficient semantic search. Best for large datasets where speed is critical."
    },
    "clip-ViT-B-16": {
        "name": "CLIP ViT-B/16",
        "repo": "sentence-transformers/clip-ViT-B-16",
        "dims": 512,
        "size_mb": 605,
        "powerpoints": ["🎯 High Accuracy", "🖼️ Better Detail", "⚖️ Balanced"],
        "description": "Better precision than B/32 by looking at smaller patches. Ideal for detailed scene understanding."
    },
    "clip-ViT-L-14": {
        "name": "CLIP ViT-L/14",
        "repo": "sentence-transformers/clip-ViT-L-14",
        "dims": 768,
        "size_mb": 1700,
        "powerpoints": ["🏆 State-of-the-Art", "🎨 Fine Detail", "🖥️ Heavy Resource"],
        "description": "Maximum accuracy and detail retrieval. Requires significant resources but provides the best results."
    }
}

# --- Logging & Storage Helpers ---
def log_event(message):
    with open("app.log", "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {message}\n")

def get_dir_size(path):
    total = 0
    if not os.path.exists(path): return 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try: total += os.path.getsize(fp)
            except: pass
    return total / (1024 * 1024) # MB

def reset_database(client, collection_name):
    try:
        client.delete_collection(collection_name)
        if os.path.exists("storage"): shutil.rmtree("storage")
        log_event(f"Database {collection_name} and storage cleared.")
        return True
    except Exception as e:
        log_event(f"Reset failed: {e}")
        return False

# --- Imports & Function Definitions (Moved to top to avoid NameErrors) ---
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
    collection_name = f"visionai_v5_{model_name.replace('-', '_').replace('.', '_')}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "ip"}
    )

def get_digested_items(collection):
    results = collection.get()
    items = []
    seen_paths = set()
    if results['ids']:
        for i in range(len(results['ids'])):
            meta = results['metadatas'][i]
            path = meta.get('path')
            if path not in seen_paths:
                items.append({
                    "id": results['ids'][i],
                    "path": path,
                    "type": meta.get('type', 'image'),
                    "name": meta.get('name', results['ids'][i])
                })
                seen_paths.add(path)
    return items

def download_model_with_progress(repo_id, size_mb):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate progress because snapshot_download doesn't have a simple callback for bytes
    # But we can use the fact that it blocks and show a realistic animation
    start_time = time.time()
    
    with st.spinner(f"Downloading {repo_id}..."):
        # Real download
        snapshot_download(repo_id=repo_id, local_dir=os.path.join(HF_CACHE, f"models--{repo_id.replace('/', '--')}"))
        
    duration = time.time() - start_time
    progress_bar.progress(100)
    status_text.success(f"Download complete in {int(duration)}s!")

def process_video(filepath, filename, model, collection, progress_bar, status_text):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        st.error(f"Could not open video {filename}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    status_text.text(f"Processing {filename} ({int(duration)}s)...")
    
    # Process 1 frame per second
    for second in range(int(duration)):
        frame_idx = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Resize for CLIP (in-memory only)
        clip_img = pil_img.resize((224, 224))
        
        # Embed
        raw_emb = model.encode(clip_img)
        embedding = (raw_emb / np.linalg.norm(raw_emb)).tolist()
        
        # Add to DB
        collection.add(
            ids=[f"{filename}_s{second}"],
            embeddings=[embedding],
            metadatas=[{
                "path": filepath,
                "timestamp": float(second),
                "type": "video",
                "name": filename
            }]
        )
        
        progress_bar.progress((second + 1) / int(duration))
    
    cap.release()
    log_event(f"Video {filename} processed ({int(duration)}s).")

def render_result(results, i, query, mode="small"):
    meta = results['metadatas'][0][i]
    res_path = meta['path']
    image_id = results['ids'][0][i]
    distance = results['distances'][0][i]
    score = 1 - distance
    res_type = meta.get('type', 'image')
    
    with st.container(border=True):
        if not os.path.exists(res_path):
            st.warning(f"File missing: {res_path}")
            return

        if res_type == 'video':
            ts = meta.get('timestamp', 0)
            if mode != "tiny":
                st.markdown(f"🎬 **{meta.get('name')}** at `{ts}s`")
            
            # Start 2s before. Use a container to avoid direct key issues if any
            start_time = max(0, int(ts) - 2)
            st.video(res_path, start_time=start_time)
            
            if mode != "tiny":
                st.caption(f"Score: `{score:.4f}` | Starts context at {start_time}s")
        else:
            if mode == "list":
                c1, c2 = st.columns([1, 4])
                with c1: st.image(res_path, use_container_width=True)
                with c2: 
                    st.markdown(f"**{image_id}** (Score: `{score:.4f}`)")
                    if st.button("🔍 Original", key=f"btn_{image_id}_{query}"):
                        st.image(res_path)
            elif mode == "large":
                st.image(res_path, use_container_width=True)
                st.markdown(f"🖼️ **{image_id}** | Score: `{score:.4f}`")
                if st.button("🔍 View Full", key=f"btn_{image_id}_{query}"):
                    st.image(res_path)
            elif mode == "tiny":
                st.image(res_path, use_container_width=True)
            else: # small/grid
                st.image(res_path, use_container_width=True)
                st.caption(f"Score: `{score:.4f}`")
                if st.button("🔍", key=f"btn_{image_id}_{query}"):
                    st.image(res_path)

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

# --- Sidebar: Navigation ---
with st.sidebar:
    st.title("🚀 VisionAI")
    page = st.radio("Navigation", [
        "📖 Guide", 
        "🔍 Search", 
        "🖼️ Digested Images", 
        "🎬 Digested Videos", 
        "📂 Digested Documents", 
        "📤 Ingest",
        "📊 Storage & Details",
        "📝 System Logs"
    ])
    
    st.divider()
    
    # Global Config at the Bottom
    st.subheader("⚙️ System Config")
    online_mode = st.toggle("Online Mode", value=False)
    
    if not online_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    model_options = list(MODEL_INFO.keys())
    selected_model_id = st.selectbox("CLIP Architecture", model_options)
    info = MODEL_INFO[selected_model_id]
    
    model_folder = f"models--{info['repo'].replace('/', '--')}"
    is_cached = os.path.exists(os.path.join(HF_CACHE, model_folder))
    
    if not is_cached:
        if st.button(f"📥 Download {info['name']}", key="manual_dl", type="primary", use_container_width=True):
            try:
                download_model_with_progress(info['repo'], info['size_mb'])
                log_event(f"Downloaded model: {info['name']}")
                load_model.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")
    else:
        st.success("✅ Model Ready")

    st.divider()
    # # 3. System Logs
    # with st.expander("📝 System Logs", expanded=False):
    #     if os.path.exists("app.log"):
    #         with open("app.log", "r", encoding="utf-8") as f:
    #             logs = f.readlines()
    #             st.code("".join(logs[-15:]), language="text")
    #     else:
    #         st.caption("No logs available yet.")

    # # 4. Storage & Details
    # with st.expander("📊 Storage & Details", expanded=False):
    #     db_size = get_dir_size("chroma_db")
    #     model_size = get_dir_size("hf_cache")
    #     storage_size = get_dir_size("storage")
        
    #     st.markdown(f"""
    #         - **Vectors (DB):** `{db_size:.2f} MB`
    #         - **Models Cache:** `{model_size:.2f} MB`
    #         - **Raw Media:** `{storage_size:.2f} MB`
    #     """)
        
    #     if st.button("🔥 Reset Collection", type="secondary", use_container_width=True):
    #         if reset_database(get_chromadb_client(), f"visionai_v5_{selected_model_id.replace('-', '_').replace('.', '_')}"):
    #             st.success("Database cleared!")
    #             st.rerun()
    #         else:
    #             st.error("Failed to reset database.")


def main():
    st.title("🖼️ VisionAI Semantic Retrieval")
    st.markdown("Search your local image library using natural language embeddings.")

    # Initialize Session State
    if "history" not in st.session_state: st.session_state.history = []
    if "ingest_key" not in st.session_state: st.session_state.ingest_key = 0

    client = get_chromadb_client()
    
    # Initialize Model & Collection
    try:
        if not is_cached:
            st.error("🚨 **System Initialization Required**")
            st.info("Follow these steps to get started:")
            st.markdown("""
                1. 🌐 **Enable Online Mode** in the sidebar.
                2. 📥 **Click the Download Button** that appears in the sidebar.
                3. ☕ **Wait** for the one-time model download to finish.
                4. 🔒 **Turn off Online Mode** if you wish to go offline.
            """)
            return
            
        with st.status(f"Initializing {selected_model_id}...", expanded=False) as status:
            model = load_model(selected_model_id)
            collection = get_collection(client, selected_model_id)
            status.update(label="System Ready", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return

    # --- Page Content Rendering ---
    if page == "📖 Guide":
        st.header("📖 Welcome to VisionAI")
        st.markdown(f"""
            <div class="model-card">
                <h4>System Status: {"✅ Ready" if is_cached else "🚨 Initialization Required"}</h4>
                <p>Model: <b>{info['name']}</b> ({info['dims']} dims)</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            ### How to Use this App:
            1. **Ingest Media**: Go to the **📤 Ingest** page to index your photos and videos.
            2. **Search**: Use the **🔍 Search** page to find media using natural language.
            3. **Library**: View all your indexed media in the **Digested** pages.
            4. **System**: Monitor storage and logs in the **📊 Details** and **📝 Logs** pages.
            
            *Tip: Use "Tiles" view for scanning large libraries, and "Details" for a closer look.*
        """)

    elif page == "🔍 Search":
        col_q, col_f, col_v, col_n = st.columns([2, 1, 1, 1])
        with col_q: query = st.text_input("Search query:", placeholder="e.g., 'a photo of a cat'")
        with col_f: media_filter = st.selectbox("Media Type", ["All", "Images", "Videos", "Documents"])
        with col_v: view_mode = st.selectbox("View Mode", ["Grid", "Tiles", "Details", "List"])
        with col_n: n_results = st.number_input("Count", 1, 50, 6)
        
        if st.session_state.history:
            with st.expander("🕒 Recent Searches"):
                for h in reversed(st.session_state.history[-5:]):
                    if st.button(f"🔍 {h}", key=f"hist_{h}"):
                        query = h
                        st.rerun()

        if query:
            if query not in st.session_state.history:
                st.session_state.history.append(query)
                log_event(f"Search query: {query}")
            with st.spinner("Searching..."):
                raw_query_emb = model.encode(query)
                query_embedding = (raw_query_emb / np.linalg.norm(raw_query_emb)).tolist()
                where_filter = {}
                if media_filter == "Images": where_filter = {"type": "image"}
                elif media_filter == "Videos": where_filter = {"type": "video"}
                results = collection.query(query_embeddings=[query_embedding], n_results=int(n_results), where=where_filter if where_filter else None)
                if results['ids'] and results['ids'][0]:
                    st.subheader(f"Results ({len(results['ids'][0])})")
                    if view_mode == "Grid":
                        cols = st.columns(3)
                        for i in range(len(results['ids'][0])):
                            with cols[i % 3]: render_result(results, i, query, mode="small")
                    elif view_mode == "Tiles":
                        cols = st.columns(5)
                        for i in range(len(results['ids'][0])):
                            with cols[i % 5]: render_result(results, i, query, mode="tiny")
                    elif view_mode == "Details":
                        for i in range(len(results['ids'][0])): render_result(results, i, query, mode="large")
                    elif view_mode == "List":
                        for i in range(len(results['ids'][0])): render_result(results, i, query, mode="list")
                else: st.info("No matches found.")

    elif page == "🖼️ Digested Images":
        library_items = get_digested_items(collection)
        imgs = [item for item in library_items if item['type'] == 'image']
        if imgs:
            cols = st.columns(4)
            for i, img in enumerate(imgs):
                with cols[i % 4]:
                    st.image(img['path'], use_container_width=True)
                    st.caption(img['name'])
        else: st.info("No images digested yet.")

    elif page == "🎬 Digested Videos":
        library_items = get_digested_items(collection)
        vids = [item for item in library_items if item['type'] == 'video']
        if vids:
            for vid in vids:
                with st.expander(f"🎬 {vid['name']}"): st.video(vid['path'])
        else: st.info("No videos digested yet.")

    elif page == "📂 Digested Documents":
        st.info("🚀 **Coming Soon!** Document indexing is under development.")

    elif page == "📤 Ingest":
        st.subheader("Ingest New Media")
        uploaded_files = st.file_uploader("Drop files here", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"], accept_multiple_files=True, key=f"uploader_{st.session_state.ingest_key}")
        if uploaded_files and st.button("Process & Index", type="primary"):
            progress_bar = st.progress(0); status_text = st.empty()
            STORAGE_DIR = "storage"; os.makedirs(STORAGE_DIR, exist_ok=True)
            existing_ids = set(collection.get()['ids']); new_count = 0
            for idx, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                if filename in existing_ids: continue
                filepath = os.path.join(STORAGE_DIR, filename)
                with open(filepath, "wb") as f: f.write(uploaded_file.getbuffer())
                try:
                    if filename.lower().endswith(('.mp4', '.avi', '.mov')): process_video(filepath, filename, model, collection, progress_bar, status_text)
                    else:
                        status_text.text(f"Processing {filename}...")
                        image = Image.open(uploaded_file).convert("RGB")
                        clip_img = image.resize((224, 224))
                        raw_emb = model.encode(clip_img)
                        embedding = (raw_emb / np.linalg.norm(raw_emb)).tolist()
                        collection.add(ids=[filename], embeddings=[embedding], metadatas=[{"path": filepath, "type": "image"}])
                    new_count += 1
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                except Exception as e: st.error(f"Error: {e}")
            st.success(f"Ingestion complete! Added {new_count} items.")
            st.session_state.ingest_key += 1
            st.rerun()

    elif page == "📊 Storage & Details":
        st.header("📊 Storage Management")
        db_size = get_dir_size("chroma_db")
        model_size = get_dir_size("hf_cache")
        storage_size = get_dir_size("storage")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vectors", f"{db_size:.2f} MB")
        c2.metric("Models", f"{model_size:.2f} MB")
        c3.metric("Media", f"{storage_size:.2f} MB")
        st.divider()
        st.subheader("Model Details")
        st.markdown(f"**Selected:** {info['name']}\n\n{info['description']}")
        st.divider()
        if st.button("🔥 Wipe Database & Storage", type="secondary"):
            if reset_database(get_chromadb_client(), f"visionai_v5_{selected_model_id.replace('-', '_').replace('.', '_')}"):
                st.success("Wiped successfully.")
                st.rerun()

    elif page == "📝 System Logs":
        st.header("📝 System Logs")
        if os.path.exists("app.log"):
            with open("app.log", "r", encoding="utf-8") as f:
                logs = f.readlines()
                st.code("".join(logs[-50:]), language="text")
        else: st.caption("No logs yet.")

if __name__ == "__main__":
    main()
