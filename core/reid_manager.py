from __future__ import annotations

from pathlib import Path

import numpy as np

from core.models import CameraConfig, LocalTrack, ReIDConfig, TrackletObservation
from core.reid_backend import ReIDStatus, create_reid_backend


class ReIDManager:
    def __init__(self, config: ReIDConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self._backend = create_reid_backend(config, project_root)
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._backend.load()
        self._loaded = True

    def status(self) -> ReIDStatus:
        self.ensure_loaded()
        if self._backend.is_available():
            return ReIDStatus(backend_name=self.config.backend, available=True)
        reason = getattr(self._backend, "load_error", None) or getattr(self._backend, "reason", None)
        return ReIDStatus(backend_name=self.config.backend, available=False, degraded_reason=reason)

    def build_tracklet_observations(
        self,
        frame_bgr: np.ndarray,
        camera_config: CameraConfig,
        local_tracks: dict[int, LocalTrack],
        *,
        timestamp: float,
        frame_index: int,
        media_time_s: float | None,
    ) -> list[TrackletObservation]:
        self.ensure_loaded()
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        ordered_track_ids: list[int] = []
        for track_id, track in local_tracks.items():
            if not self._qualifies(track, camera_config):
                continue
            bbox = track.current_bbox_xyxy or track.last_bbox_xyxy_for_matching
            if bbox is None:
                continue
            ordered_track_ids.append(track_id)
            items.append((frame_bgr, bbox))

        embeddings = self._backend.embed_batch(items) if items else []
        embedding_by_track_id = dict(zip(ordered_track_ids, embeddings))

        observations: list[TrackletObservation] = []
        for track_id, track in local_tracks.items():
            embedding = embedding_by_track_id.get(track_id, track.appearance_descriptor)
            if embedding:
                track.appearance_descriptor = embedding
            bbox = track.current_bbox_xyxy or track.last_bbox_xyxy_for_matching
            if bbox is None:
                continue
            observations.append(
                TrackletObservation(
                    camera_id=camera_config.camera_id,
                    tracker_track_id=track.local_track_id,
                    timestamp=timestamp,
                    bbox_xyxy=bbox,
                    ground_anchor_world=track.ground_anchor_world,
                    ground_anchor_image=track.ground_anchor_image,
                    confidence=track.confidence,
                    appearance_embedding=list(track.appearance_descriptor),
                    frame_index=frame_index,
                    media_time_s=media_time_s,
                    entry_edge=track.last_entry_edge,
                    exit_edge=track.last_exit_edge,
                )
            )
        return observations

    def _qualifies(self, track: LocalTrack, camera_config: CameraConfig) -> bool:
        if not self.config.enabled:
            return False
        min_confidence = min(self.config.min_confidence, self.config.overlap_reid_min_confidence)
        if track.confidence < min_confidence:
            return False
        bbox = track.current_bbox_xyxy or track.last_bbox_xyxy_for_matching
        if bbox is None:
            return False
        min_height_px = min(self.config.min_bbox_height_px, self.config.overlap_reid_min_bbox_height_px)
        return (bbox[3] - bbox[1]) >= min_height_px
