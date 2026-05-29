# AGENTS

## Purpose
Booth analytics on a shared 2D world map. The app synchronizes camera feeds, tracks people locally, deduplicates them only inside calibrated camera overlap, and computes booth dwell/occupancy metrics.

## Architecture Overview
- `core/streaming.py`: per-camera worker. Reads frames, runs detection/tracking, emits local tracks and camera identity tracks.
- `core/camera_identity.py`: stabilizes tracker tracklets inside one camera.
- `core/overlap_dedup.py`: overlap-only cross-camera dedup for counting, not full event-wide identity.
- `core/metrics.py`: booth visit session engine. Computes occupancy, visits, avg/median dwell, peak occupancy.
- `core/multi_camera_runtime.py`: orchestrates workers, sync, overlap graph, dedup, analytics, and operator frames.
- `ui/main_window.py`: operator shell. Camera grid left, booth map right.

## Calibration Model
- Shared anchors are the source of truth for camera-to-world homography.
- Canonical persisted coverage is `coverage_polygon_image`.
- World coverage is derived from image coverage on load/runtime.
- Overlap is built only from recomputed sanitized world coverage.
- Viewport is auto-fit from valid camera coverage plus zones.
- World coordinates are relative by default. True meters are not inferred from video alone.

## Key Runtime Modules
- `core/calibration.py`: homography, projection, coverage recomputation, viewport.
- `core/coverage_mapping.py`: semi-auto coverage proposal from a frame.
- `core/camera_overlap.py`: overlap graph from derived coverage.
- `core/detection.py`: detector helpers and overlays.
- `core/reid_backend.py` / `core/reid_manager.py`: appearance embeddings used only to support overlap dedup.
- `ui/multi_camera_calibration_dialog.py`: anchors + image-space coverage editing.
- `ui/map_view.py`: map, zones, camera coverage, overlap, active booth occupancy markers.

## Important Config Files
- `config/project.json`: top-level project state.
- `config/cameras/*.json`: per-camera source, tracker, calibration, coverage.
- `config/venue.json`: map image and booth zones.

## Startup / Run
- Preferred: `.venv/bin/python main.py`
- Fallback: `python main.py` if the environment already has required GUI/video/ML deps.

## Testing
- `.venv/bin/python -m unittest discover -s tests`
- `python -m compileall core ui tests main.py app/main.py`

## Debugging Checklist
- If booth dwell looks wrong:
  - inspect `core/metrics.py`
  - check zone geometry and `zone_entry_min_duration_s` / `zone_exit_grace_s`
- If overlap double-counting happens:
  - inspect `core/overlap_dedup.py`
  - verify overlap graph and camera coverage
  - confirm embeddings are available and stable
- If overlap looks wrong:
  - inspect homography RMSE/warnings
  - inspect `coverage_polygon_image`, not just world overlay
- If MP4 sync drifts:
  - inspect `core/media_sync.py`
  - confirm `session_sync_mode == all_file_strict`

## Current Limitations
- Anchor calibration is still manual.
- Coverage auto-detection is assistive, not authoritative.
- Local tracking quality still affects booth dwell accuracy.
- Cross-camera dedup is intentionally limited to overlap and conservative thresholds.

## Editing Conventions
- Derived geometry must be recomputed from canonical state.
- Use overlap dedup only to prevent double-counting, not to rebuild event-wide identity.
- Keep operator UI booth-centric, not identity-centric.
