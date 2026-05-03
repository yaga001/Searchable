# Implementation Plan - Optimized Desktop App (Thin Client)

The goal is to create a small, fast-loading desktop application that downloads heavy assets (models and environment components) on-demand, rather than bundling them.

## User Review Required

> [!IMPORTANT]
> **Size Optimization**: To keep the initial download small, I will:
> 1. Use **PyTorch CPU** instead of the full GPU version (saves ~1.5 GB).
> 2. **Exclude Models**: The `hf_cache` will NOT be bundled. The app will download models when the user first toggles "Online Mode".
> 3. **External Persistence**: All data (`chroma_db`, `images`, `hf_cache`) will live in the user's local app data folder, making the executable purely a functional "shell".

## Proposed Changes

### 1. Optimize Requirements
Create a `requirements-desktop.txt` optimized for size, using CPU-only versions of heavy libraries.

#### [NEW] [requirements-desktop.txt](file:///e:/ANTIGRAVITY/requirements-desktop.txt)
- `torch --index-url https://download.pytorch.org/whl/cpu`
- `torchvision --index-url https://download.pytorch.org/whl/cpu`
- Other essential Streamlit and Search dependencies.

### 2. Update Application Logic for "First Run"
Modify `app.py` to detect if models are missing and guide the user to download them.

#### [MODIFY] [app.py](file:///e:/ANTIGRAVITY/app.py)
- Check if `hf_cache` contains the selected model.
- If missing, display a "Download Required" warning with a button to enable Online Mode.

### 3. Build Configuration
Configure `streamlit-desktop-app` to strictly bundle only the code and essential UI assets.

#### [RUN]
```powershell
# Exclude data directories during build
.\venv\Scripts\streamlit-desktop-app build app.py --name "VisionAI" --exclude "hf_cache,chroma_db,images,venv"
```

### 4. Installer (Optional Suggestion)
I will provide a small `setup.ps1` script that acts as an "Online Installer" which:
1. Downloads the executable.
2. Checks for Python.
3. Initializes the local directories.

## Verification Plan

### Automated Tests
- Build the app and verify the `.exe` size is significantly smaller (aiming for < 500MB).
- Verify that the `.exe` starts even if `hf_cache` is deleted.

### Manual Verification
- Run the app, select a model, and verify the "Download" workflow triggers correctly.
- Confirm that once downloaded, the model persists across restarts.
