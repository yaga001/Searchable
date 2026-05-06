# Implementation Plan - Advanced UI & System Management

Enhance the VisionAI app with precise video playback, flexible layout options, search history, and storage/log management.

## Proposed Changes

### UI & UX Improvements

#### [MODIFY] [app.py](file:///e:/ANTIGRAVITY/app.py)
- **Video Fixes**:
    - Force `st.video` to jump to `start_time` by using a dynamic `key` (e.g., `f"vid_{res_path}_{ts}_{query}"`).
    - Constrain video width using CSS or `st.container` with fixed dimensions.
- **Search Layouts**:
    - Add a "View Mode" selector in the Search tab: `Grid`, `Tile`, `Details`, `List`.
    - Implement conditional rendering for each mode.
- **Search History**:
    - Store queries in `st.session_state.history`.
    - Show a "Recent Searches" section in the search tab.
- **Ingest UX**:
    - Use `st.empty()` or state clearing to remove files from the UI after processing.

### Sidebar System Management

#### [MODIFY] [app.py](file:///e:/ANTIGRAVITY/app.py)
- **Logs Tab**:
    - Add a "System Logs" expander in the sidebar.
    - Implement a simple logging mechanism that reads the last 20 lines of a log file.
- **Storage Details Tab**:
    - Add a "Storage Management" expander.
    - Calculate and display sizes of `hf_cache`, `chroma_db`, and `storage`.
    - Add a "🔥 Reset Database" button with a confirmation prompt.
- **Download Button**: 
    - Re-position the download button directly under the model card.

## Verification Plan

### Manual Verification
1.  **Video Timestamp**:
    - Search for a term that appears in a video.
    - Verify it plays from ~2s before the match.
    - Search again for a different term and verify the player jumps to the *new* timestamp.
2.  **View Modes**:
    - Toggle between Grid, Tile, and List modes.
    - Verify the layout changes as expected.
3.  **Storage/Logs**:
    - Check sidebar for folder sizes.
    - Verify that indexing actions appear in the sidebar logs.
4.  **Reset DB**:
    - Use the reset button.
    - Verify the "Digested" tabs are cleared.
