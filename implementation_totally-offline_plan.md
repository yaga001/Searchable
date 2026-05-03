# Implementation Plan - Fix Dependency and Enhanced Offline/Online UI

The goal is to resolve the `torchvision` dependency issue and significantly enhance the Streamlit UI with an internet connectivity toggle, detailed model metadata, and a premium design.

## User Review Required

> [!IMPORTANT]
> **Internet Toggle**: I will implement a sidebar toggle to switch between "Offline Mode" and "Online Mode". 
> - **Offline Mode**: Blocks all external connections to HuggingFace, ensuring privacy and speed by using local cache.
> - **Online Mode**: Allows downloading new models or checking for updates.

## Proposed Changes

### 1. Fix Dependencies
Ensure `torchvision` is installed and compatible with `torch`.

#### [MODIFY] [requirements.txt](file:///e:/ANTIGRAVITY/requirements.txt)
- Pin compatible versions of `torch` and `torchvision`.

### 2. Enhanced Streamlit Application
Update `app.py` with a premium UI and new features.

#### [MODIFY] [app.py](file:///e:/ANTIGRAVITY/app.py)
- **Connectivity Toggle**: Add a sidebar toggle for Offline/Online mode.
- **Model Metadata**: Add a registry of CLIP models with:
    - **Dimensions**: (e.g., 512 for B-32)
    - **Memory Footprint**: Estimated VRAM/RAM usage.
    - **Strengths**: What the model is best at (e.g., "Fast retrieval", "High accuracy").
- **Visual Enhancements**: 
    - Use `st.status` for model loading progress.
    - Implement a "Model Card" UI to show model limits and "Power Points" (capabilities).
    - Add custom CSS for glassmorphism and a modern dark theme.

### 3. Ingestion Script (Optional Sync)
Keep `ingest.py` simple but add a flag for offline mode for consistency.

#### [MODIFY] [ingest.py](file:///e:/ANTIGRAVITY/ingest.py)
- Add `--offline` flag.

## Verification Plan

### Automated Tests
- Verify `torchvision` installation: `pip show torchvision`.
- Test model loading with the toggle in both states.

### Manual Verification
- **Aesthetics Check**: Ensure the new "Model Cards" and progress bars look premium.
- **Offline Validation**: Turn on "Offline Mode" and ensure the app doesn't hang or error when loading a cached model.
