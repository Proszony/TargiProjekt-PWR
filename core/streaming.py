from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import av
from PySide6.QtCore import QObject, Signal, Slot

from core.detection import YoloPersonDetector, annotate_frame, qimage_from_bgr
from core.fusion import GlobalFusionEngine
from core.metrics import AnalyticsEngine
from core.models import AnalyticsSnapshot, CameraConfig, LocalTrack, VenueMapConfig
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
    frame_ready = Signal(object)
    analytics_ready = Signal(object)
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
        model_path: str = "models/yolo26m.pt",
        confidence: float = 0.25,
        inference_size: int = 640,
    ) -> None:
        super().__init__()
        self._config_lock = threading.Lock()
        self.camera_config = camera_config
        self.venue_map = venue_map
        self.statistics_service = statistics_service
        self._running = False
        self._container: Optional[av.container.InputContainer] = None
        resolved_model_path = camera_config.detector_model_path or model_path
        self._detector = YoloPersonDetector(
            resolved_model_path,
            confidence=confidence,
            inference_size=inference_size,
            use_augmentation=camera_config.detector_use_augmentation,
        )
        self._tracker = SimpleWorldTracker(
            track_timeout_s=camera_config.track_timeout_s,
            max_world_distance_m=camera_config.tracker_max_world_distance_m,
            max_image_distance_px=camera_config.tracker_max_image_distance_px,
            max_missed_frames=camera_config.tracker_max_missed_frames,
            min_iou=camera_config.tracker_min_iou,
            anchor_weight=camera_config.tracker_anchor_weight,
            iou_weight=camera_config.tracker_iou_weight,
            confidence_weight=camera_config.tracker_confidence_weight,
        )
        self._tracker_adapter = UltralyticsTrackerAdapter(
            resolved_model_path,
            confidence=confidence,
            inference_size=inference_size,
            use_augmentation=camera_config.detector_use_augmentation,
        )
        self._fusion = GlobalFusionEngine()
        self._analytics = AnalyticsEngine(
            venue_map=venue_map,
            zone_entry_min_duration_s=camera_config.zone_entry_min_duration_s,
            return_threshold_s=camera_config.return_threshold_s,
        )
        self._detection_enabled = True
        self._session_started = False
        self._runtime_tracker_signature: tuple[object, ...] | None = None

    @Slot()
    def run(self) -> None:
        self._running = True
        self.started_listening.emit()
        smoothed_fps = 0.0
        last_frame_t = time.perf_counter()
        last_snapshot_persisted_at = 0.0

        while self._running:
            try:
                with self._config_lock:
                    camera_config = CameraConfig.from_dict(self.camera_config.to_dict())
                self._container = self._open_source(camera_config)
                if not self._session_started:
                    self.statistics_service.start_session(
                        started_at=time.time(),
                        source_type=camera_config.source_type,
                        source_label=self._source_label(camera_config),
                        camera_id=camera_config.camera_id,
                    )
                    self._session_started = True
                self.status_changed.emit(f"Connected: {self._source_label(camera_config)}")

                for frame in self._container.decode(video=0):
                    if not self._running:
                        break

                    now = time.perf_counter()
                    delta = max(now - last_frame_t, 1e-6)
                    last_frame_t = now
                    smoothed_fps = 0.1 * (1.0 / delta) + 0.9 * smoothed_fps
                    self.fps_update.emit(smoothed_fps)

                    timestamp = time.time()
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    frame_bgr = frame_rgb[:, :, ::-1].copy()

                    with self._config_lock:
                        camera_config = CameraConfig.from_dict(self.camera_config.to_dict())
                        venue_map = VenueMapConfig.from_dict(self.venue_map.to_dict())
                        if camera_config.source_type == "udp":
                            camera_config.udp_url = camera_config.source_value
                        camera_config.tracker_config_path = self._materialize_tracker_config(camera_config)
                        self._detector.set_model_path(camera_config.detector_model_path)
                        self._detector.use_augmentation = camera_config.detector_use_augmentation
                        self._tracker_adapter.set_model_path(camera_config.detector_model_path)
                        self._tracker_adapter.use_augmentation = camera_config.detector_use_augmentation
                        self._tracker_adapter.confidence = self._detector.confidence
                        self._tracker_adapter.inference_size = self._detector.inference_size
                        self._tracker.track_timeout_s = camera_config.track_timeout_s
                        self._tracker.max_world_distance_m = camera_config.tracker_max_world_distance_m
                        self._tracker.max_image_distance_px = camera_config.tracker_max_image_distance_px
                        self._tracker.max_missed_frames = camera_config.tracker_max_missed_frames
                        self._tracker.min_iou = camera_config.tracker_min_iou
                        self._tracker.anchor_weight = camera_config.tracker_anchor_weight
                        self._tracker.iou_weight = camera_config.tracker_iou_weight
                        self._tracker.confidence_weight = camera_config.tracker_confidence_weight
                        self._analytics.venue_map = venue_map
                        self._analytics.zone_entry_min_duration_s = camera_config.zone_entry_min_duration_s
                        self._analytics.return_threshold_s = camera_config.return_threshold_s

                    local_tracks: dict[int, LocalTrack] = {}
                    expired_tracks: list[LocalTrack] = []
                    if self._detection_enabled:
                        if camera_config.tracker_backend in {"botsort", "bytetrack"}:
                            detections = self._tracker_adapter.track(frame_bgr, timestamp, camera_config)
                        elif camera_config.tracker_backend == "legacy":
                            detections = self._detector.detect(frame_bgr, timestamp, camera_config)
                        else:
                            raise ValueError(
                                f"Unsupported tracker backend '{camera_config.tracker_backend}'. Use botsort or bytetrack."
                            )
                        local_tracks, expired_tracks = self._tracker.update(
                            camera_id=camera_config.camera_id,
                            timestamp=timestamp,
                            detections=detections,
                        )
                    global_tracks = self._fusion.update(timestamp, local_tracks, expired_tracks)
                    analytics_snapshot = self._analytics.update(timestamp, global_tracks)

                    for track in local_tracks.values():
                        track.active_zone_id = None
                        track.display_track_id = None
                        global_key = f"{track.camera_id}:{track.local_track_id}"
                        for global_track in global_tracks.values():
                            if global_key in global_track.member_local_tracks:
                                track.active_zone_id = global_track.current_zone_id
                                track.display_track_id = global_track.global_track_id
                                break

                    if timestamp - last_snapshot_persisted_at >= 1.0:
                        self.statistics_service.record_snapshot(
                            analytics_snapshot,
                            venue_map,
                            global_tracks=global_tracks,
                        )
                        last_snapshot_persisted_at = timestamp

                    annotated = annotate_frame(frame_bgr, local_tracks, camera_config, venue_map)
                    self.frame_ready.emit(qimage_from_bgr(annotated))
                    self.analytics_ready.emit(analytics_snapshot)

                if not self._running:
                    break
                if camera_config.source_type == "file":
                    if camera_config.loop_file:
                        self.status_changed.emit("Reached end of file, restarting...")
                        continue
                    self.status_changed.emit("Playback finished")
                    self._running = False
                    break
                self.status_changed.emit("Stream lost, reconnecting...")
            except Exception as exc:
                if not self._running:
                    break
                self.status_changed.emit("Error opening or decoding stream; retrying...")
                tracker_backend = getattr(camera_config, "tracker_backend", "unknown")
                self.error_occurred.emit(
                    f"{exc} (tracker={tracker_backend}; switch to ByteTrack if BoT-SORT fails on this setup)"
                )
            finally:
                if self._container is not None:
                    try:
                        self._container.close()
                    except Exception:
                        pass
                    self._container = None

            if self._running:
                time.sleep(0.5 if camera_config.source_type == "file" else 1.0)

        self.statistics_service.finish_session(time.time())
        self._session_started = False
        self.status_changed.emit("Stopped")
        self.stopped_listening.emit()

    @Slot()
    def stop(self) -> None:
        self._running = False

    def update_configs(self, camera_config: CameraConfig, venue_map: VenueMapConfig) -> None:
        with self._config_lock:
            self.camera_config = camera_config
            self.venue_map = venue_map

    def set_detection_enabled(self, enabled: bool) -> None:
        self._detection_enabled = enabled

    def set_confidence(self, confidence: float) -> None:
        self._detector.confidence = confidence
        self._tracker_adapter.confidence = confidence

    def set_inference_size(self, inference_size: int) -> None:
        self._detector.inference_size = inference_size
        self._tracker_adapter.inference_size = inference_size

    def set_detector_model_path(self, model_path: str) -> None:
        self._detector.set_model_path(model_path)
        self._tracker_adapter.set_model_path(model_path)

    def set_detector_augmentation(self, enabled: bool) -> None:
        self._detector.use_augmentation = enabled
        self._tracker_adapter.use_augmentation = enabled

    def _open_source(self, camera_config: CameraConfig) -> av.container.InputContainer:
        if camera_config.source_type == "file":
            source_path = Path(camera_config.source_value).expanduser()
            return av.open(str(source_path))
        camera_config.udp_url = camera_config.source_value
        self.status_changed.emit(f"Listening on {camera_config.source_value}")
        return av.open(
            camera_config.source_value,
            options=STREAM_OPTIONS,
            timeout=(STREAM_OPEN_TIMEOUT_S, STREAM_READ_TIMEOUT_S),
        )

    @staticmethod
    def _source_label(camera_config: CameraConfig) -> str:
        if camera_config.source_type == "file":
            return Path(camera_config.source_value).name or camera_config.source_value
        return camera_config.source_value

    def _materialize_tracker_config(self, camera_config: CameraConfig) -> str:
        signature = (
            camera_config.tracker_backend,
            camera_config.tracker_reid_enabled,
            camera_config.tracker_track_buffer,
            camera_config.tracker_match_thresh,
            camera_config.tracker_new_track_thresh,
            camera_config.tracker_proximity_thresh,
            camera_config.tracker_appearance_thresh,
            round(self._tracker_adapter.confidence, 4),
        )
        runtime_dir = Path.cwd() / "data" / "runtime_trackers"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = runtime_dir / f"{camera_config.camera_id}_{camera_config.tracker_backend}.yaml"
        if signature == self._runtime_tracker_signature and runtime_path.exists():
            return str(runtime_path)

        backend = camera_config.tracker_backend if camera_config.tracker_backend in {"botsort", "bytetrack"} else "botsort"
        lines = [
            f"tracker_type: {backend}",
            f"track_high_thresh: {max(self._tracker_adapter.confidence, 0.1):.3f}",
            "track_low_thresh: 0.1",
            f"new_track_thresh: {camera_config.tracker_new_track_thresh:.3f}",
            f"track_buffer: {camera_config.tracker_track_buffer}",
            f"match_thresh: {camera_config.tracker_match_thresh:.3f}",
            "fuse_score: True",
        ]
        if backend == "botsort":
            lines.extend(
                [
                    "gmc_method: sparseOptFlow",
                    f"proximity_thresh: {camera_config.tracker_proximity_thresh:.3f}",
                    f"appearance_thresh: {camera_config.tracker_appearance_thresh:.3f}",
                    f"with_reid: {'True' if camera_config.tracker_reid_enabled else 'False'}",
                    "model: auto",
                ]
            )
        runtime_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._runtime_tracker_signature = signature
        return str(runtime_path)
