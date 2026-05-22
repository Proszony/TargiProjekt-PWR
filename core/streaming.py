from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

import av
from PySide6.QtCore import QObject, Signal, Slot

from core import runtime_defaults as rd
from core.camera_identity import CameraIdentityEngine
from core.calibration import recompute_camera_coverage
from core.coverage_mapping import propose_coverage_polygon_image
from core.detection import YoloPersonDetector, annotate_frame, qimage_from_bgr
from core.media_time import resolve_media_time, resolve_stream_fps
from core.model_catalog import resolve_detector_model_spec
from core.models import CameraConfig, CameraTrackingPacket, IdentityConfig, LocalTrack, PlaybackSyncConfig, ReIDConfig, VenueMapConfig
from core.reid_manager import ReIDManager
from core.statistics_service import StatisticsService
from core.tracker_adapter import UltralyticsTrackerAdapter
from core.tracking import SimpleWorldTracker

STREAM_OPTIONS = {
    "fflags": "nobuffer",
    "flags": "low_delay",
    "strict": "experimental",
    "analyzeduration": "0",
}
STREAM_OPEN_TIMEOUT_S = 1.0
STREAM_READ_TIMEOUT_S = 1.0


class CameraPipelineWorker(QObject):
    frame_ready = Signal(str, int, object)
    camera_packet_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    started_listening = Signal()
    stopped_listening = Signal()
    fps_update = Signal(float)

    def __init__(
        self,
        camera_config: CameraConfig,
        venue_map: VenueMapConfig,
        statistics_service: StatisticsService,
        project_root: Path,
        playback_sync_config: PlaybackSyncConfig | None = None,
        session_sync_mode: str = "all_live_unsynced",
        reid_config: ReIDConfig | None = None,
        identity_config: IdentityConfig | None = None,
        file_playback_started_wall_time: float | None = None,
        model_path: str = rd.DEFAULT_DETECTOR_MODEL_PATH,
        confidence: float = rd.DEFAULT_DETECTOR_CONFIDENCE,
        inference_size: int = rd.DEFAULT_DETECTOR_INFERENCE_SIZE,
        on_frame_ready: Callable[[str, int, object], None] | None = None,
        on_camera_packet_ready: Callable[[object], None] | None = None,
        on_status_changed: Callable[[str], None] | None = None,
        on_error_occurred: Callable[[str], None] | None = None,
        on_started_listening: Callable[[], None] | None = None,
        on_stopped_listening: Callable[[], None] | None = None,
        on_fps_update: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self._config_lock = threading.Lock()
        self.camera_config = camera_config
        self.venue_map = venue_map
        self.statistics_service = statistics_service
        self.project_root = project_root
        self.playback_sync_config = playback_sync_config or PlaybackSyncConfig()
        self.session_sync_mode = session_sync_mode
        self.reid_config = reid_config or ReIDConfig()
        self.identity_config = identity_config or IdentityConfig()
        self.file_playback_started_wall_time = file_playback_started_wall_time
        self._running = False
        self._container: Optional[av.container.InputContainer] = None
        self._source_fps: float | None = None
        self._frame_index = 0
        self._dropped_frames = 0
        self._tracker_config_path = ""
        resolved_model_path = resolve_detector_model_spec(
            self.project_root,
            camera_config.detector_model_path or model_path,
        )
        self._detector = YoloPersonDetector(
            resolved_model_path,
            confidence=confidence,
            inference_size=inference_size,
            use_augmentation=rd.DEFAULT_DETECTOR_AUGMENTATION,
        )
        self._tracker = SimpleWorldTracker(
            track_timeout_s=rd.DEFAULT_TRACK_TIMEOUT_S,
            max_world_distance_m=rd.DEFAULT_TRACKER_MAX_WORLD_DISTANCE_M,
            max_image_distance_px=rd.DEFAULT_TRACKER_MAX_IMAGE_DISTANCE_PX,
            max_missed_frames=rd.DEFAULT_TRACKER_MAX_MISSED_FRAMES,
            min_iou=rd.DEFAULT_TRACKER_MIN_IOU,
            anchor_weight=rd.DEFAULT_TRACKER_ANCHOR_WEIGHT,
            iou_weight=rd.DEFAULT_TRACKER_IOU_WEIGHT,
            confidence_weight=rd.DEFAULT_TRACKER_CONFIDENCE_WEIGHT,
        )
        self._tracker_adapter = UltralyticsTrackerAdapter(
            resolved_model_path,
            confidence=confidence,
            inference_size=inference_size,
            use_augmentation=rd.DEFAULT_DETECTOR_AUGMENTATION,
        )
        self._reid_manager = ReIDManager(self.reid_config, self.project_root)
        self._camera_identity = CameraIdentityEngine(
            camera_config.camera_id,
            self.identity_config,
            self.reid_config,
        )
        self._on_frame_ready = on_frame_ready
        self._on_camera_packet_ready = on_camera_packet_ready
        self._on_status_changed = on_status_changed
        self._on_error_occurred = on_error_occurred
        self._on_started_listening = on_started_listening
        self._on_stopped_listening = on_stopped_listening
        self._on_fps_update = on_fps_update
        self._detection_enabled = True
        self._runtime_tracker_signature: tuple[object, ...] | None = None

    @Slot()
    def run(self) -> None:
        self._running = True
        self._publish_started_listening()
        smoothed_fps = 0.0
        last_frame_t = time.perf_counter()
        last_snapshot_persisted_at = 0.0

        while self._running:
            try:
                with self._config_lock:
                    camera_config = CameraConfig.from_dict(self.camera_config.to_dict())
                self._container = self._open_source(camera_config)
                self._prepare_source_timing()
                self._publish_status_changed(f"Connected: {self._source_label(camera_config)}")

                for frame in self._container.decode(video=0):
                    if not self._running:
                        break

                    now = time.perf_counter()
                    delta = max(now - last_frame_t, 1e-6)
                    last_frame_t = now
                    smoothed_fps = 0.1 * (1.0 / delta) + 0.9 * smoothed_fps
                    self._publish_fps_update(smoothed_fps)

                    wall_time_s = time.time()
                    media_time_s = (
                        resolve_media_time(frame, self._source_fps, self._frame_index)
                        if camera_config.source_type == "file"
                        else None
                    )
                    self._pace_file_frame(media_time_s)
                    if self._should_skip_file_frame(media_time_s):
                        self._frame_index += 1
                        self._dropped_frames += 1
                        continue

                    timestamp = media_time_s if media_time_s is not None else wall_time_s
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    frame_bgr = frame_rgb[:, :, ::-1].copy()
                    processing_started_at = time.perf_counter()

                    with self._config_lock:
                        camera_config = CameraConfig.from_dict(self.camera_config.to_dict())
                        venue_map = VenueMapConfig.from_dict(self.venue_map.to_dict())
                        self._tracker_config_path = self._materialize_tracker_config(camera_config)
                        resolved_model_spec = resolve_detector_model_spec(
                            self.project_root,
                            camera_config.detector_model_path or rd.DEFAULT_DETECTOR_MODEL_PATH,
                        )
                        self._detector.set_model_path(resolved_model_spec)
                        self._detector.use_augmentation = rd.DEFAULT_DETECTOR_AUGMENTATION
                        self._tracker_adapter.set_model_path(resolved_model_spec)
                        self._tracker_adapter.use_augmentation = rd.DEFAULT_DETECTOR_AUGMENTATION
                        self._tracker_adapter.confidence = self._detector.confidence
                        self._tracker_adapter.inference_size = self._detector.inference_size
                        self._tracker.track_timeout_s = rd.DEFAULT_TRACK_TIMEOUT_S
                        self._tracker.max_world_distance_m = rd.DEFAULT_TRACKER_MAX_WORLD_DISTANCE_M
                        self._tracker.max_image_distance_px = rd.DEFAULT_TRACKER_MAX_IMAGE_DISTANCE_PX
                        self._tracker.max_missed_frames = rd.DEFAULT_TRACKER_MAX_MISSED_FRAMES
                        self._tracker.min_iou = rd.DEFAULT_TRACKER_MIN_IOU
                        self._tracker.anchor_weight = rd.DEFAULT_TRACKER_ANCHOR_WEIGHT
                        self._tracker.iou_weight = rd.DEFAULT_TRACKER_IOU_WEIGHT
                        self._tracker.confidence_weight = rd.DEFAULT_TRACKER_CONFIDENCE_WEIGHT
                        self._camera_identity.identity_config = self.identity_config
                        self._camera_identity.reid_config = self.reid_config

                    local_tracks: dict[int, LocalTrack] = {}
                    expired_tracks: list[LocalTrack] = []
                    if self._detection_enabled:
                        detections = self._tracker_adapter.track(
                            frame_bgr,
                            timestamp,
                            camera_config,
                            self._tracker_config_path,
                        )
                        local_tracks, expired_tracks = self._tracker.update(
                            camera_id=camera_config.camera_id,
                            timestamp=timestamp,
                            detections=detections,
                        )
                    camera_config.frame_width = frame_bgr.shape[1]
                    camera_config.frame_height = frame_bgr.shape[0]
                    self._refresh_camera_coverage(camera_config, frame_bgr)
                    self._annotate_track_geometry(local_tracks, frame_bgr.shape[1], frame_bgr.shape[0])
                    self._annotate_expired_tracks(expired_tracks, frame_bgr.shape[1], frame_bgr.shape[0])
                    tracklet_observations = self._reid_manager.build_tracklet_observations(
                        frame_bgr,
                        camera_config,
                        local_tracks,
                        timestamp=timestamp,
                        frame_index=self._frame_index,
                        media_time_s=media_time_s,
                    )
                    camera_identity_tracks, expired_camera_identity_tracks, identity_debug_records = (
                        self._camera_identity.update(
                            timestamp,
                            local_tracks,
                            expired_tracks,
                            tracklet_observations,
                        )
                    )
                    reid_status = self._reid_manager.status()

                    annotated = annotate_frame(
                        frame_bgr,
                        local_tracks,
                        camera_config,
                        venue_map,
                        render_timestamp=timestamp,
                    )
                    processing_latency_s = time.perf_counter() - processing_started_at
                    self._publish_frame_ready(
                        camera_config.camera_id,
                        self._frame_index,
                        qimage_from_bgr(annotated),
                    )
                    self._publish_camera_packet_ready(
                        CameraTrackingPacket(
                            camera_id=camera_config.camera_id,
                            timestamp=timestamp,
                            wall_time_s=wall_time_s,
                            media_time_s=media_time_s,
                            frame_index=self._frame_index,
                            source_kind="file" if camera_config.source_type == "file" else "live",
                            source_fps=self._source_fps,
                            sync_ready=(media_time_s is not None) if camera_config.source_type == "file" else True,
                            dropped_frame_count=self._dropped_frames,
                            processing_latency_s=processing_latency_s,
                            tracklet_observations=tracklet_observations,
                            local_tracks=dict(local_tracks),
                            expired_tracks=list(expired_tracks),
                            camera_identity_tracks=camera_identity_tracks,
                            expired_camera_identity_tracks=expired_camera_identity_tracks,
                            identity_debug_records=identity_debug_records,
                            reid_backend_ready=reid_status.available,
                            frame_size=(frame_bgr.shape[1], frame_bgr.shape[0]),
                            coverage_polygon_image=(
                                list(camera_config.coverage_polygon_image)
                                if camera_config.coverage_polygon_image
                                else None
                            ),
                            coverage_polygon_world_raw=(
                                list(camera_config.coverage_polygon_world_raw)
                                if camera_config.coverage_polygon_world_raw
                                else None
                            ),
                            coverage_polygon_world=(
                                list(camera_config.coverage_polygon_world)
                                if camera_config.coverage_polygon_world
                                else None
                            ),
                            coverage_auto_generated=camera_config.coverage_auto_generated,
                            coverage_confidence=camera_config.coverage_confidence,
                            coverage_warning_text=camera_config.coverage_warning_text,
                            fps=smoothed_fps,
                            status_text=self._source_label(camera_config),
                        )
                    )
                    self._frame_index += 1

                if not self._running:
                    break
                if camera_config.source_type == "file":
                    if camera_config.loop_file:
                        self._frame_index = 0
                        self._dropped_frames = 0
                        self._publish_status_changed("Reached end of file, restarting...")
                        continue
                    self._publish_status_changed("Playback finished")
                    self._running = False
                    break
                self._publish_status_changed("Stream lost, reconnecting...")
            except Exception as exc:
                if not self._running:
                    break
                self._publish_status_changed("Error opening or decoding stream; retrying...")
                self._publish_error_occurred(str(exc))
            finally:
                if self._container is not None:
                    try:
                        self._container.close()
                    except Exception:
                        pass
                    self._container = None

            if self._running:
                time.sleep(0.5 if camera_config.source_type == "file" else 1.0)

        self._publish_status_changed("Stopped")
        self._publish_stopped_listening()

    @Slot()
    def stop(self) -> None:
        self._running = False

    def update_configs(self, camera_config: CameraConfig, venue_map: VenueMapConfig) -> None:
        with self._config_lock:
            self.camera_config = camera_config
            self.venue_map = venue_map
            self._camera_identity.camera_id = camera_config.camera_id

    def set_playback_sync(
        self,
        playback_sync_config: PlaybackSyncConfig,
        file_playback_started_wall_time: float | None,
        session_sync_mode: str | None = None,
    ) -> None:
        self.playback_sync_config = playback_sync_config
        self.file_playback_started_wall_time = file_playback_started_wall_time
        if session_sync_mode is not None:
            self.session_sync_mode = session_sync_mode

    def set_identity_configs(self, reid_config: ReIDConfig, identity_config: IdentityConfig) -> None:
        self.reid_config = reid_config
        self.identity_config = identity_config
        self._reid_manager = ReIDManager(reid_config, self.project_root)
        self._camera_identity.identity_config = identity_config
        self._camera_identity.reid_config = reid_config

    def set_detection_enabled(self, enabled: bool) -> None:
        self._detection_enabled = enabled

    def set_confidence(self, confidence: float) -> None:
        self._detector.confidence = confidence
        self._tracker_adapter.confidence = confidence

    def set_inference_size(self, inference_size: int) -> None:
        self._detector.inference_size = inference_size
        self._tracker_adapter.inference_size = inference_size

    def _refresh_camera_coverage(
        self,
        camera_config: CameraConfig,
        frame_bgr,
    ) -> None:
        if camera_config.homography_image_to_world is None:
            return
        if not camera_config.coverage_polygon_image:
            proposal = propose_coverage_polygon_image(
                frame_bgr,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
            )
            if proposal.polygon_image:
                camera_config.coverage_polygon_image = list(proposal.polygon_image)
                camera_config.coverage_auto_generated = True
                camera_config.coverage_confidence = proposal.confidence
                camera_config.coverage_warning_text = " | ".join(proposal.warnings)
        coverage_result = recompute_camera_coverage(camera_config)
        camera_config.coverage_polygon_world_raw = coverage_result.raw_polygon_world or None
        camera_config.coverage_polygon_world = coverage_result.sanitized_polygon_world or None
        combined_warnings = [
            part
            for part in [
                camera_config.calibration_warning_text,
                camera_config.coverage_warning_text,
                " | ".join(coverage_result.warnings) if coverage_result.warnings else "",
            ]
            if part
        ]
        camera_config.coverage_warning_text = " | ".join(coverage_result.warnings)
        camera_config.calibration_warning_text = " | ".join(dict.fromkeys(combined_warnings))
        camera_config.calibration_valid = bool(camera_config.homography_image_to_world) and coverage_result.is_valid

    def set_detector_model_path(self, model_path: str) -> None:
        resolved = resolve_detector_model_spec(self.project_root, model_path or rd.DEFAULT_DETECTOR_MODEL_PATH)
        self._detector.set_model_path(resolved)
        self._tracker_adapter.set_model_path(resolved)

    def set_detector_augmentation(self, enabled: bool) -> None:
        self._detector.use_augmentation = rd.DEFAULT_DETECTOR_AUGMENTATION
        self._tracker_adapter.use_augmentation = rd.DEFAULT_DETECTOR_AUGMENTATION

    def _open_source(self, camera_config: CameraConfig) -> av.container.InputContainer:
        self._source_fps = None
        self._frame_index = 0
        self._dropped_frames = 0
        if camera_config.source_type == "file":
            source_path = Path(camera_config.source_value).expanduser()
            if not source_path.is_absolute():
                source_path = (self.project_root / source_path).resolve()
            return av.open(str(source_path))
        self._publish_status_changed(f"Listening on {camera_config.source_value}")
        return av.open(
            camera_config.source_value,
            options=STREAM_OPTIONS,
            timeout=(STREAM_OPEN_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
        )

    def _publish_frame_ready(self, camera_id: str, frame_index: int, image: object) -> None:
        self.frame_ready.emit(camera_id, frame_index, image)
        if self._on_frame_ready is not None:
            self._on_frame_ready(camera_id, frame_index, image)

    def _publish_camera_packet_ready(self, packet: object) -> None:
        self.camera_packet_ready.emit(packet)
        if self._on_camera_packet_ready is not None:
            self._on_camera_packet_ready(packet)

    def _publish_status_changed(self, status_text: str) -> None:
        self.status_changed.emit(status_text)
        if self._on_status_changed is not None:
            self._on_status_changed(status_text)

    def _publish_error_occurred(self, message: str) -> None:
        self.error_occurred.emit(message)
        if self._on_error_occurred is not None:
            self._on_error_occurred(message)

    def _publish_started_listening(self) -> None:
        self.started_listening.emit()
        if self._on_started_listening is not None:
            self._on_started_listening()

    def _publish_stopped_listening(self) -> None:
        self.stopped_listening.emit()
        if self._on_stopped_listening is not None:
            self._on_stopped_listening()

    def _publish_fps_update(self, fps: float) -> None:
        self.fps_update.emit(fps)
        if self._on_fps_update is not None:
            self._on_fps_update(fps)

    @staticmethod
    def _source_label(camera_config: CameraConfig) -> str:
        if camera_config.source_type == "file":
            return Path(camera_config.source_value).name or camera_config.source_value
        return camera_config.source_value

    def _prepare_source_timing(self) -> None:
        if self._container is None:
            return
        try:
            stream = self._container.streams.video[0]
        except (IndexError, AttributeError):
            self._source_fps = None
            return
        self._source_fps = resolve_stream_fps(stream.average_rate) or resolve_stream_fps(
            getattr(stream, "base_rate", None)
        )

    def _should_skip_file_frame(self, media_time_s: float | None) -> bool:
        if media_time_s is None or self.file_playback_started_wall_time is None:
            return False
        lag_s = time.perf_counter() - (self.file_playback_started_wall_time + media_time_s)
        return lag_s > self.playback_sync_config.late_frame_drop_threshold_s

    def _pace_file_frame(self, media_time_s: float | None) -> None:
        if media_time_s is None or self.file_playback_started_wall_time is None:
            return
        target_wall_time = self.file_playback_started_wall_time + media_time_s
        sleep_s = target_wall_time - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    def _materialize_tracker_config(self, camera_config: CameraConfig) -> str:
        signature = (
            rd.DEFAULT_TRACKER_BACKEND,
            rd.DEFAULT_TRACKER_REID_ENABLED,
            rd.DEFAULT_TRACKER_TRACK_BUFFER,
            rd.DEFAULT_TRACKER_MATCH_THRESH,
            rd.DEFAULT_TRACKER_NEW_TRACK_THRESH,
            rd.DEFAULT_TRACKER_PROXIMITY_THRESH,
            rd.DEFAULT_TRACKER_APPEARANCE_THRESH,
            round(self._tracker_adapter.confidence, 4),
        )
        runtime_dir = Path.cwd() / "data" / "runtime_trackers"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = runtime_dir / f"{camera_config.camera_id}_{rd.DEFAULT_TRACKER_BACKEND}.yaml"
        if signature == self._runtime_tracker_signature and runtime_path.exists():
            return str(runtime_path)

        backend = rd.DEFAULT_TRACKER_BACKEND
        lines = [
            f"tracker_type: {backend}",
            f"track_high_thresh: {max(self._tracker_adapter.confidence, 0.1):.3f}",
            "track_low_thresh: 0.05",
            f"new_track_thresh: {rd.DEFAULT_TRACKER_NEW_TRACK_THRESH:.3f}",
            f"track_buffer: {rd.DEFAULT_TRACKER_TRACK_BUFFER}",
            f"match_thresh: {rd.DEFAULT_TRACKER_MATCH_THRESH:.3f}",
            "fuse_score: True",
        ]
        if backend == "botsort":
            lines.extend(
                [
                    "gmc_method: sparseOptFlow",
                    f"proximity_thresh: {rd.DEFAULT_TRACKER_PROXIMITY_THRESH:.3f}",
                    f"appearance_thresh: {rd.DEFAULT_TRACKER_APPEARANCE_THRESH:.3f}",
                    f"with_reid: {'True' if rd.DEFAULT_TRACKER_REID_ENABLED else 'False'}",
                    "model: auto",
                ]
            )
        runtime_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._runtime_tracker_signature = signature
        return str(runtime_path)

    @staticmethod
    def _annotate_track_geometry(
        local_tracks: dict[int, LocalTrack],
        frame_width: int,
        frame_height: int,
    ) -> None:
        for track in local_tracks.values():
            CameraPipelineWorker._annotate_single_track(track, frame_width, frame_height)

    @staticmethod
    def _annotate_expired_tracks(
        expired_tracks: list[LocalTrack],
        frame_width: int,
        frame_height: int,
    ) -> None:
        for track in expired_tracks:
            CameraPipelineWorker._annotate_single_track(track, frame_width, frame_height)

    @staticmethod
    def _annotate_single_track(track: LocalTrack, frame_width: int, frame_height: int) -> None:
        bbox = track.current_bbox_xyxy or track.last_bbox_xyxy_for_matching
        if bbox is None:
            return
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        track.bbox_center_image = center
        edge_name, edge_score = CameraPipelineWorker._edge_proximity(center, frame_width, frame_height)
        track.edge_proximity_score = edge_score
        if edge_name is not None:
            track.last_entry_edge = edge_name
            track.last_exit_edge = edge_name

    @staticmethod
    def _edge_proximity(
        center: tuple[float, float],
        frame_width: int,
        frame_height: int,
    ) -> tuple[str | None, float]:
        margin_x = max(frame_width * 0.10, 48.0)
        margin_y = max(frame_height * 0.10, 48.0)
        distances = {
            "left": center[0],
            "right": frame_width - center[0],
            "top": center[1],
            "bottom": frame_height - center[1],
        }
        edge_name = min(distances, key=distances.get)
        nearest_distance = distances[edge_name]
        threshold = margin_x if edge_name in {"left", "right"} else margin_y
        if nearest_distance > threshold:
            return None, 0.0
        score = 1.0 - min(nearest_distance / max(threshold, 1e-6), 1.0)
        return edge_name, score
