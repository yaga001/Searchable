# Implementation Plan - Video Semantic Search

Add semantic search capabilities for video files by extracting frames at 1 FPS, indexing them with CLIP, and providing a video player that jumps to the relevant timestamp.

## Proposed Changes

### Core Logic & UI

#### [MODIFY] [app.py](file:///e:/ANTIGRAVITY/app.py)
- Update `get_collection` to handle a dedicated video collection or add a type filter.
- Update "Ingest" tab:
    - Support video file formats (`mp4`, `avi`, `mov`).
    - Implement `process_video` function:
        - Use OpenCV to extract 1 frame per second.
        - Resize frames to 224x224.
        - Vectorize each frame with CLIP.
        - Store in ChromaDB with `path`, `timestamp`, and `frame_idx` metadata.
- Update "Search" tab:
    - Display video results using `st.video`.
    - Automatically set `start_time` to `timestamp - 2` seconds.
    - Display the specific timestamp where the match was found.

#### [MODIFY] [ingest.py](file:///e:/ANTIGRAVITY/ingest.py)
- Add video processing logic to the CLI ingestion script for parity with the UI.

## Verification Plan

### Automated Tests
- None (UI-based verification).

### Manual Verification
1.  **Video Ingestion**:
    - Upload a short video (e.g., 10-20 seconds).
    - Verify the progress bar shows frame-by-frame vectorization.
    - Check that the data is saved in ChromaDB.
2.  **Video Search**:
    - Search for an object or action present in the video.
    - Verify that the video result appears.
    - Click play and verify it starts ~2 seconds before the matched event.
3.  **Image Parity**:
    - Ensure image search still works as expected.
