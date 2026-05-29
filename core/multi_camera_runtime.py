from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen

from core import runtime_defaults as rd
from core.calibration import compute_world_viewport
from core.camera_overlap import build_camera_overlap_graph
from core.distributed_server import DistributedRuntimeServer
from core.heatmap import HeatmapAccumulator
from core.media_sync import MultiCameraMediaSynchronizer
from core.metrics import AnalyticsEngine
from core.models import (
    AnalyticsSnapshot,
    CameraConfig,
    CameraOverlapGraph,
    CameraTrackingPacket,
    HeatmapSnapshot,
    MapPresence,
    MultiCameraRuntimeSnapshot,
    Point,
    ProjectConfig,
    SynchronizedCameraFrameSet,
    WorldViewport,
)
from core.statistics_service import StatisticsService
from core.streaming import CameraPipelineWorker


class MultiCameraPipelineManager(QObject):
    camera_frame_ready = Signal(str, object)
    camera_status_changed = Signal(str, str)
    camera_error = Signal(str, str)
    camera_fps_changed = Signal(str, float)
    analytics_ready = Signal(object)
    runtime_snapshot_ready = Signal(object)
    started = Signal()
    stopped = Signal()

    def __init__(
        self,
        project_config: ProjectConfig,
        statistics_service: StatisticsService,
        project_root: Path,
    ) -> None:
        super().__init__()
        self.project_config = project_config
        self.statistics_service = statistics_service
        self.project_root = project_root
        self._workers: dict[str, CameraPipelineWorker] = {}
        self._threads: dict[str, QThread] = {}
        self._packets: dict[str, CameraTrackingPacket] = {}
        self._frame_images: dict[str, dict[int, object]] = {}
        self._latest_remote_preview_images: dict[str, QImage] = {}
        self._latest_remote_preview_frame_indices: dict[str, int] = {}
        self._latest_remote_labels_by_camera: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {}
        self._labeled_remote_preview_cache: dict[
            str,
            tuple[int, tuple[tuple[tuple[int, int, int, int], str], ...], QImage],
        ] = {}
        self._analytics = AnalyticsEngine(
            venue_map=project_config.venue_map,
            zone_entry_min_duration_s=project_config.analytics.zone_entry_min_duration_s,
            zone_exit_grace_s=project_config.analytics.zone_exit_grace_s,
        )
        self._heatmap = HeatmapAccumulator(
            enabled=project_config.analytics.heatmap_enabled,
            sample_interval_s=project_config.analytics.heatmap_sample_interval_s,
            grid_columns=project_config.analytics.heatmap_grid_columns,
            min_rows=project_config.analytics.heatmap_min_rows,
            max_rows=project_config.analytics.heatmap_max_rows,
        )
        self._last_heatmap_snapshot: HeatmapSnapshot | None = None
        self._last_snapshot_persisted_at = 0.0
        self._running = False
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._process_media_sync_tick)
        self._media_synchronizer: MultiCameraMediaSynchronizer | None = None
        self._session_sync_mode = "all_live_unsynced"
        self._session_started_at_unix_s: float | None = None
        self._file_playback_started_wall_time: float | None = None
        self._remote_server: DistributedRuntimeServer | None = None
        self._overlap_graph_cache_key: tuple[object, ...] | None = None
        self._overlap_graph_cache: CameraOverlapGraph | None = None
        self._heatmap_viewport_cache_key: tuple[object, ...] | None = None
        self._heatmap_viewport_cache: WorldViewport | None = None

    def update_project_config(self, project_config: ProjectConfig) -> None:
        self.project_config = ProjectConfig.from_dict(project_config.to_dict())
        self._invalidate_geometry_cache()
        self._analytics.venue_map = self.project_config.venue_map
        self._analytics.zone_entry_min_duration_s = self.project_config.analytics.zone_entry_min_duration_s
        self._analytics.zone_exit_grace_s = self.project_config.analytics.zone_exit_grace_s
        self._configure_heatmap()
        camera_lookup = {camera.camera_id: camera for camera in self.project_config.cameras}
        stale_camera_ids = set(self._packets) - set(camera_lookup)
        for camera_id in stale_camera_ids:
            self._packets.pop(camera_id, None)
            self._frame_images.pop(camera_id, None)
            self._latest_remote_preview_images.pop(camera_id, None)
            self._latest_remote_preview_frame_indices.pop(camera_id, None)
            self._latest_remote_labels_by_camera.pop(camera_id, None)
            self._labeled_remote_preview_cache.pop(camera_id, None)
        active_local_camera_ids = {
            camera.camera_id
            for camera in self.project_config.cameras
            if camera.enabled and camera.runtime_mode == "local"
        }
        active_remote_cameras = [
            camera
            for camera in self.project_config.cameras
            if camera.enabled and camera.runtime_mode == "remote"
        ]
        removed_camera_ids = set(self._workers) - active_local_camera_ids
        for camera_id in removed_camera_ids:
            worker = self._workers.get(camera_id)
            if worker is not None:
                worker.stop()
            self._packets.pop(camera_id, None)
            self._frame_images.pop(camera_id, None)
            self._latest_remote_preview_images.pop(camera_id, None)
            self._latest_remote_preview_frame_indices.pop(camera_id, None)
            self._latest_remote_labels_by_camera.pop(camera_id, None)
            self._labeled_remote_preview_cache.pop(camera_id, None)
        for camera_id, worker in self._workers.items():
            camera = camera_lookup.get(camera_id)
            if camera is None:
                continue
            worker.update_configs(camera, self.project_config.venue_map)
            worker.set_playback_sync(
                self.project_config.playback_sync,
                self._file_playback_started_wall_time,
                self._session_sync_mode,
            )
            worker.set_detection_enabled(camera.enabled)
            worker.set_detector_model_path(camera.detector_model_path)
            worker.set_detector_augmentation(camera.detector_use_augmentation)
        if self._running:
            for camera in self.project_config.cameras:
                if (
                    camera.camera_id not in self._workers
                    and camera.enabled
                    and camera.runtime_mode == "local"
                ):
                    self._start_camera_worker(camera)
            if active_remote_cameras:
                if self._remote_server is None:
                    self._start_remote_server()
                    if self._remote_server is not None:
                        self._remote_server.start_session(
                            self._session_sync_mode,
                            self._session_started_at_unix_s,
                        )
                if self._remote_server is not None:
                    self._remote_server.update_project_config(self.project_config)
                for camera in active_remote_cameras:
                    self.camera_status_changed.emit(camera.camera_id, "waiting for worker")
            elif self._remote_server is not None:
                self._remote_server.stop_session()
                self._remote_server.stop()
                self._remote_server = None
        elif self._remote_server is not None:
            self._remote_server.update_project_config(self.project_config)

    def start_all(self) -> None:
        if self._running:
            return
        self._running = True
        self._packets.clear()
        self._analytics.reset()
        self._configure_heatmap()
        self._heatmap.reset(self._cached_heatmap_viewport())
        self._last_heatmap_snapshot = self._heatmap.snapshot(0.0)
        self._last_snapshot_persisted_at = 0.0
        self._frame_images.clear()
        self._latest_remote_preview_images.clear()
        self._latest_remote_preview_frame_indices.clear()
        self._latest_remote_labels_by_camera.clear()
        self._labeled_remote_preview_cache.clear()
        self._session_started_at_unix_s = time.time()
        active_cameras = [camera for camera in self.project_config.cameras if camera.enabled]
        local_cameras = [camera for camera in active_cameras if camera.runtime_mode == "local"]
        remote_cameras = [camera for camera in active_cameras if camera.runtime_mode == "remote"]
        source_labels = [camera.source_value for camera in active_cameras]
        camera_ids = [camera.camera_id for camera in active_cameras]
        self._session_sync_mode = self._determine_session_sync_mode(
            active_cameras,
            self.project_config.playback_sync.enabled_for_file_sources,
        )
        self._file_playback_started_wall_time = self.worker_file_playback_started_wall_time(
            self._session_sync_mode,
            self._session_started_at_unix_s,
        )
        self._media_synchronizer = None
        if active_cameras:
            self.statistics_service.start_session(
                started_at=time.time(),
                source_type="multi",
                source_label=", ".join(source_labels),
                camera_id=",".join(camera_ids),
            )
        else:
            self._running = False
            self.stopped.emit()
            return
        if self._session_sync_mode == "all_file_strict":
            self._media_synchronizer = MultiCameraMediaSynchronizer(
                camera_ids,
                self.project_config.playback_sync,
            )
            timer_interval_ms = max(int(round(1000.0 / max(self.project_config.playback_sync.target_fps, 1.0))), 1)
            self._sync_timer.start(timer_interval_ms)
        else:
            self._sync_timer.stop()
        if remote_cameras:
            self._start_remote_server()
        for camera in local_cameras:
            self._start_camera_worker(camera)
        for camera in remote_cameras:
            self.camera_status_changed.emit(camera.camera_id, "waiting for worker")
        if self._remote_server is not None:
            self._remote_server.start_session(self._session_sync_mode, self._session_started_at_unix_s)
        self.started.emit()

    @staticmethod
    def worker_file_playback_started_wall_time(
        session_sync_mode: str,
        session_started_at_unix_s: float | None = None,
    ) -> float | None:
        if not (session_sync_mode.startswith("all_file") or session_sync_mode == "single_file_realtime"):
            return None
        if session_sync_mode == "single_file_realtime":
            return time.perf_counter()
        if session_started_at_unix_s is None:
            return time.perf_counter()
        return time.perf_counter() + (session_started_at_unix_s - time.time())

    def stop_all(self) -> None:
        if not self._running:
            return
        self._running = False
        self._sync_timer.stop()
        if self._remote_server is not None:
            self._remote_server.stop_session()
        for worker in list(self._workers.values()):
            worker.stop()
        if self._remote_server is not None:
            self._remote_server.stop()
            self._remote_server = None
        self._finish_stop_if_possible()

    def update_project(self, project_config: ProjectConfig) -> None:
        self.project_config = project_config
        self._invalidate_geometry_cache()
        self._analytics.venue_map = project_config.venue_map
        self._analytics.zone_entry_min_duration_s = project_config.analytics.zone_entry_min_duration_s
        self._analytics.zone_exit_grace_s = project_config.analytics.zone_exit_grace_s
        self._configure_heatmap()
        for camera in project_config.cameras:
            worker = self._workers.get(camera.camera_id)
            if worker is not None:
                worker.update_configs(camera, project_config.venue_map)
                worker.set_playback_sync(
                    project_config.playback_sync,
                    self._file_playback_started_wall_time,
                    self._session_sync_mode,
                )
        if self._remote_server is not None:
            self._remote_server.update_project_config(project_config)

    def _start_camera_worker(self, camera_config: CameraConfig) -> None:
        worker_thread = QThread(self)
        worker = CameraPipelineWorker(
            camera_config=camera_config,
            venue_map=self.project_config.venue_map,
            statistics_service=self.statistics_service,
            project_root=self.project_root,
            playback_sync_config=self.project_config.playback_sync,
            session_sync_mode=self._session_sync_mode,
            file_playback_started_wall_time=self._file_playback_started_wall_time,
            confidence=rd.DEFAULT_DETECTOR_CONFIDENCE,
            inference_size=rd.DEFAULT_DETECTOR_INFERENCE_SIZE,
        )
        worker.moveToThread(worker_thread)
        worker_thread.started.connect(worker.run)
        worker.frame_ready.connect(self._handle_camera_frame)
        worker.camera_packet_ready.connect(self._handle_camera_packet)
        worker.status_changed.connect(lambda text, camera_id=camera_config.camera_id: self.camera_status_changed.emit(camera_id, text))
        worker.error_occurred.connect(lambda text, camera_id=camera_config.camera_id: self.camera_error.emit(camera_id, text))
        worker.fps_update.connect(lambda fps, camera_id=camera_config.camera_id: self.camera_fps_changed.emit(camera_id, fps))
        worker.stopped_listening.connect(worker_thread.quit)
        worker_thread.finished.connect(lambda camera_id=camera_config.camera_id: self._cleanup_camera_worker(camera_id))
        worker.set_detection_enabled(camera_config.enabled)
        worker.set_playback_sync(
            self.project_config.playback_sync,
            self._file_playback_started_wall_time,
            self._session_sync_mode,
        )
        self._workers[camera_config.camera_id] = worker
        self._threads[camera_config.camera_id] = worker_thread
        worker_thread.start()

    @Slot(str, int, object)
    def _handle_camera_frame(self, camera_id: str, frame_index: int, image: object) -> None:
        frame_bucket = self._frame_images.setdefault(camera_id, {})
        frame_bucket[frame_index] = image

    @Slot(object)
    def _handle_camera_packet(self, packet: object) -> None:
        if not isinstance(packet, CameraTrackingPacket):
            return
        self._packets[packet.camera_id] = packet
        self.camera_fps_changed.emit(packet.camera_id, packet.fps)
        if self._session_sync_mode == "all_file_strict" and self._media_synchronizer is not None:
            self._media_synchronizer.add_packet(packet)
            return
        self._process_packet_group(
            timestamp=packet.timestamp,
            camera_packets=dict(self._packets),
            missing_cameras=self._current_missing_cameras(),
            dropped_frames_by_camera={
                camera_id: camera_packet.dropped_frame_count
                for camera_id, camera_packet in self._packets.items()
            },
            sync_drift_by_camera={
                camera_id: 0.0
                for camera_id in self._packets
            },
            session_media_time_s=packet.media_time_s,
        )

    @Slot(object)
    def _handle_remote_packet(self, packet: object) -> None:
        self._handle_camera_packet(packet)

    @Slot(object)
    def _handle_remote_preview_frame(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        camera_id = str(payload.get("camera_id", ""))
        frame_index = int(payload.get("frame_index", 0))
        jpeg_bytes = payload.get("jpeg_bytes", b"")
        if not camera_id or not isinstance(jpeg_bytes, (bytes, bytearray)):
            return
        image = QImage.fromData(bytes(jpeg_bytes), "JPEG")
        if image.isNull():
            return
        self._latest_remote_preview_images[camera_id] = image
        self._latest_remote_preview_frame_indices[camera_id] = frame_index
        self._emit_labeled_remote_preview_if_changed(camera_id)

    @Slot(str, str)
    def _handle_remote_camera_status(self, camera_id: str, status_text: str) -> None:
        self.camera_status_changed.emit(camera_id, status_text)

    @Slot(str, str)
    def _handle_remote_camera_error(self, camera_id: str, message: str) -> None:
        self.camera_error.emit(camera_id, message)

    @Slot()
    def _process_media_sync_tick(self) -> None:
        if (
            not self._running
            or self._session_sync_mode != "all_file_strict"
            or self._media_synchronizer is None
        ):
            return
        frame_set = self._media_synchronizer.next_frame_set(time.perf_counter())
        if frame_set is None:
            return
        self._process_synchronized_frame_set(frame_set)

    def _process_synchronized_frame_set(self, frame_set: SynchronizedCameraFrameSet) -> None:
        self._process_packet_group(
            timestamp=frame_set.media_time_s,
            camera_packets=frame_set.camera_packets,
            missing_cameras=sorted(set(frame_set.missing_cameras) | set(self._current_missing_cameras())),
            dropped_frames_by_camera=frame_set.dropped_packets_by_camera,
            sync_drift_by_camera=frame_set.drift_by_camera_s,
            session_media_time_s=frame_set.media_time_s,
        )

    def _process_packet_group(
        self,
        *,
        timestamp: float,
        camera_packets: dict[str, CameraTrackingPacket],
        missing_cameras: list[str],
        dropped_frames_by_camera: dict[str, int],
        sync_drift_by_camera: dict[str, float],
        session_media_time_s: float | None,
    ) -> None:
        for camera in self.project_config.cameras:
            packet = camera_packets.get(camera.camera_id)
            if packet is None:
                continue
            if packet.coverage_polygon_image:
                camera.coverage_polygon_image = list(packet.coverage_polygon_image)
            camera.coverage_auto_generated = packet.coverage_auto_generated
            camera.coverage_confidence = packet.coverage_confidence
            camera.coverage_warning_text = packet.coverage_warning_text
            camera.frame_width, camera.frame_height = packet.frame_size
            camera.coverage_polygon_world_raw = (
                list(packet.coverage_polygon_world_raw)
                if packet.coverage_polygon_world_raw
                else None
            )
            camera.coverage_polygon_world = (
                list(packet.coverage_polygon_world)
                if packet.coverage_polygon_world
                else None
            )
        camera_lookup = {camera.camera_id: camera for camera in self.project_config.cameras}
        overlap_graph = self._cached_overlap_graph()
        map_presences = self._build_local_map_presences(camera_packets)
        analytics_snapshot = self._analytics.update(timestamp, map_presences)
        heatmap_snapshot = self._heatmap.update(
            timestamp,
            analytics_snapshot.active_map_presences,
            self._cached_heatmap_viewport(),
        )
        analytics_snapshot.heatmap_snapshot = heatmap_snapshot
        self._last_heatmap_snapshot = heatmap_snapshot
        labels_by_camera = self._build_operator_labels(camera_packets)
        camera_lookup = {camera.camera_id: camera for camera in self.project_config.cameras}
        self._latest_remote_labels_by_camera = {
            camera_id: list(labels_by_camera.get(camera_id, []))
            for camera_id, camera in camera_lookup.items()
            if camera.runtime_mode == "remote"
        }
        for camera_id in self._latest_remote_preview_images:
            self._emit_labeled_remote_preview_if_changed(camera_id)
        runtime_snapshot = MultiCameraRuntimeSnapshot(
            timestamp=timestamp,
            analytics_snapshot=analytics_snapshot,
            camera_packets=dict(camera_packets),
            overlap_graph=overlap_graph,
            session_sync_mode=self._session_sync_mode,
            session_media_time_s=session_media_time_s,
            sync_drift_by_camera_s=dict(sync_drift_by_camera),
            dropped_frames_by_camera=dict(dropped_frames_by_camera),
            missing_cameras=list(missing_cameras),
            active_analytics_track_count=len(map_presences),
            active_map_presence_count=len(map_presences),
        )
        if timestamp - self._last_snapshot_persisted_at >= 1.0:
            self.statistics_service.record_snapshot(
                analytics_snapshot,
                self.project_config.venue_map,
            )
            self._last_snapshot_persisted_at = timestamp
        self.analytics_ready.emit(analytics_snapshot)
        self.runtime_snapshot_ready.emit(runtime_snapshot)
        self._emit_operator_frames(camera_packets, labels_by_camera)

    @staticmethod
    def _determine_session_sync_mode(active_cameras: list[CameraConfig], file_sync_enabled: bool = True) -> str:
        if not active_cameras:
            return "all_live_unsynced"
        source_types = {camera.source_type for camera in active_cameras}
        if source_types == {"file"}:
            if len(active_cameras) == 1:
                return "single_file_realtime"
            return "all_file_strict" if file_sync_enabled else "all_file_unsynced"
        if source_types == {"udp"}:
            return "all_live_unsynced"
        return "mixed_unsynced"

    def _drop_stale_frame_images(self, camera_id: str, keep_from_index: int) -> None:
        frame_bucket = self._frame_images.get(camera_id)
        if not frame_bucket:
            return
        for frame_index in [index for index in frame_bucket if index < keep_from_index]:
            frame_bucket.pop(frame_index, None)

    def _emit_operator_frames(
        self,
        camera_packets: dict[str, CameraTrackingPacket],
        labels_by_camera: dict[str, list[tuple[tuple[int, int, int, int], str]]],
    ) -> None:
        remote_camera_ids = {
            camera.camera_id for camera in self.project_config.cameras if camera.runtime_mode == "remote"
        }
        for camera_id, packet in camera_packets.items():
            if camera_id in remote_camera_ids:
                continue
            image = self._frame_images.get(camera_id, {}).get(packet.frame_index)
            if not isinstance(image, QImage):
                continue
            labeled = self._draw_operator_labels(image, labels_by_camera.get(camera_id, []))
            self.camera_frame_ready.emit(camera_id, labeled)
            self._drop_stale_frame_images(camera_id, keep_from_index=packet.frame_index - 1)

    def _build_local_map_presences(
        self,
        camera_packets: dict[str, CameraTrackingPacket],
    ) -> list[MapPresence]:
        camera_lookup = {camera.camera_id: camera for camera in self.project_config.cameras}
        presences: list[MapPresence] = []
        for camera_id, packet in camera_packets.items():
            camera = camera_lookup.get(camera_id)
            if camera is None or not camera.enabled or not camera.calibration_valid:
                continue
            for local_track_id, track in packet.local_tracks.items():
                if not track.active:
                    continue
                point = track.smoothed_ground_anchor_world or track.ground_anchor_world
                if point is None:
                    continue
                presence_id = f"{camera_id}:L{local_track_id}"
                presences.append(
                    MapPresence(
                        presence_id=presence_id,
                        world_point=point,
                        source_camera_ids=[camera_id],
                        source_camera_person_ids={camera_id: presence_id},
                        merged_for_counting=False,
                        confidence=track.confidence,
                        dedup_mode="local_only",
                        first_seen_ts=track.first_seen_ts,
                        last_seen_ts=track.last_seen_ts,
                    )
                )
        presences.sort(key=lambda item: item.presence_id)
        return presences

    @staticmethod
    def _build_operator_labels(
        camera_packets: dict[str, CameraTrackingPacket],
    ) -> dict[str, list[tuple[tuple[int, int, int, int], str]]]:
        labels_by_camera: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {}
        for camera_id, packet in camera_packets.items():
            for local_track_id, track in packet.local_tracks.items():
                bbox = track.current_bbox_xyxy or track.last_bbox_xyxy_for_matching
                if bbox is None:
                    continue
                labels_by_camera.setdefault(camera_id, []).append(
                    (bbox, f"{camera_id}|L{local_track_id}")
                )
        return labels_by_camera

    def _emit_labeled_remote_preview_if_changed(self, camera_id: str) -> None:
        image = self._latest_remote_preview_images.get(camera_id)
        frame_index = self._latest_remote_preview_frame_indices.get(camera_id)
        if image is None or frame_index is None:
            return
        labels_signature = self._operator_labels_signature(
            self._latest_remote_labels_by_camera.get(camera_id, [])
        )
        cached = self._labeled_remote_preview_cache.get(camera_id)
        if (
            cached is not None
            and cached[0] == frame_index
            and cached[1] == labels_signature
        ):
            return
        labeled = self._draw_operator_labels(image, list(labels_signature))
        self._labeled_remote_preview_cache[camera_id] = (frame_index, labels_signature, labeled)
        self.camera_frame_ready.emit(camera_id, labeled)

    @staticmethod
    def _operator_labels_signature(
        labels: list[tuple[tuple[int, int, int, int], str]],
    ) -> tuple[tuple[tuple[int, int, int, int], str], ...]:
        normalized = [
            ((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), str(label))
            for bbox, label in labels
            if label
        ]
        return tuple(sorted(normalized, key=lambda item: (item[1], item[0])))

    @staticmethod
    def _draw_operator_labels(
        image: QImage,
        labels: list[tuple[tuple[int, int, int, int], str]],
    ) -> QImage:
        labels = [(bbox, label) for bbox, label in labels if label]
        if not labels:
            return image
        annotated = image.copy()
        painter = QPainter(annotated)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        font = QFont()
        font.setPointSizeF(10.0)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        for bbox, label in labels:
            x1, y1, _x2, _y2 = bbox
            text_rect = metrics.boundingRect(label)
            chip_width = text_rect.width() + 12
            chip_height = text_rect.height() + 8
            chip_x = max(4, x1)
            chip_y = y1 - chip_height - 6
            if chip_y < 4:
                chip_y = y1 + 6
            background = QColor("#0f1720")
            background.setAlpha(210)
            border = QColor("#22c55e")
            painter.setPen(QPen(border, 1.25))
            painter.setBrush(background)
            painter.drawRoundedRect(chip_x, chip_y, chip_width, chip_height, 6, 6)
            painter.setPen(QColor("#f8fafc"))
            baseline_y = chip_y + metrics.ascent() + 4
            painter.drawText(chip_x + 6, baseline_y, label)

        painter.end()
        return annotated

    def _cleanup_camera_worker(self, camera_id: str) -> None:
        worker = self._workers.pop(camera_id, None)
        if worker is not None:
            worker.deleteLater()
        thread = self._threads.pop(camera_id, None)
        if thread is not None:
            thread.deleteLater()
        self._packets.pop(camera_id, None)
        self._frame_images.pop(camera_id, None)
        self._latest_remote_preview_images.pop(camera_id, None)
        self._latest_remote_preview_frame_indices.pop(camera_id, None)
        self._latest_remote_labels_by_camera.pop(camera_id, None)
        self._labeled_remote_preview_cache.pop(camera_id, None)
        if self._running and self._workers:
            return
        self._finish_stop_if_possible()

    def _start_remote_server(self) -> None:
        if self._remote_server is not None:
            self._remote_server.update_project_config(self.project_config)
            return
        server = DistributedRuntimeServer(self.project_config)
        server.camera_packet_received.connect(self._handle_remote_packet)
        server.preview_frame_received.connect(self._handle_remote_preview_frame)
        server.camera_status_changed.connect(self._handle_remote_camera_status)
        server.camera_error.connect(self._handle_remote_camera_error)
        server.start()
        self._remote_server = server

    def _current_missing_cameras(self) -> list[str]:
        missing = set()
        if self._remote_server is not None:
            missing.update(self._remote_server.unavailable_camera_ids())
        return sorted(missing)

    def _finish_stop_if_possible(self) -> None:
        if self._running:
            return
        if self._threads:
            return
        self.statistics_service.finish_session_with_heatmap(time.time(), self._last_heatmap_snapshot)
        self.stopped.emit()

    def _configure_heatmap(self) -> None:
        analytics = self.project_config.analytics
        self._heatmap.configure(
            enabled=analytics.heatmap_enabled,
            sample_interval_s=analytics.heatmap_sample_interval_s,
            grid_columns=analytics.heatmap_grid_columns,
            min_rows=analytics.heatmap_min_rows,
            max_rows=analytics.heatmap_max_rows,
        )

    def _invalidate_geometry_cache(self) -> None:
        self._overlap_graph_cache_key = None
        self._overlap_graph_cache = None
        self._heatmap_viewport_cache_key = None
        self._heatmap_viewport_cache = None

    def _cached_overlap_graph(self) -> CameraOverlapGraph:
        cache_key = self._overlap_graph_key()
        if self._overlap_graph_cache is None or self._overlap_graph_cache_key != cache_key:
            self._overlap_graph_cache = build_camera_overlap_graph(
                self.project_config.cameras,
                self.project_config.overlap_dedup,
            )
            self._overlap_graph_cache_key = cache_key
        return self._overlap_graph_cache

    def _cached_heatmap_viewport(self) -> WorldViewport:
        cache_key = self._heatmap_viewport_key()
        if self._heatmap_viewport_cache is None or self._heatmap_viewport_cache_key != cache_key:
            self._heatmap_viewport_cache = self._build_heatmap_viewport()
            self._heatmap_viewport_cache_key = cache_key
        return self._heatmap_viewport_cache

    def _build_heatmap_viewport(self) -> WorldViewport:
        zone_polygons = [
            zone.polygon_world
            for zone in self.project_config.venue_map.zones
            if len(zone.polygon_world) >= 3
        ]
        if zone_polygons or self.project_config.venue_map.manual_viewport_override is not None:
            return compute_world_viewport(
                [],
                self.project_config.venue_map.zones,
                padding_ratio=0.12,
                manual_override=self.project_config.venue_map.manual_viewport_override,
            )
        anchor_points = [anchor.world_point for anchor in self.project_config.shared_anchors]
        if anchor_points:
            return self._viewport_from_points(anchor_points)
        return compute_world_viewport(
            self.project_config.cameras,
            self.project_config.venue_map.zones,
            manual_override=self.project_config.venue_map.manual_viewport_override,
        )

    @staticmethod
    def _viewport_from_points(points: list[Point]) -> WorldViewport:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        pad_x = span_x * 0.12
        pad_y = span_y * 0.12
        return WorldViewport(
            min_x=min_x - pad_x,
            min_y=min_y - pad_y,
            max_x=max_x + pad_x,
            max_y=max_y + pad_y,
        )

    def _overlap_graph_key(self) -> tuple[object, ...]:
        overlap = self.project_config.overlap_dedup
        return (
            tuple(
                (
                    camera.camera_id,
                    camera.calibration_valid,
                    self._point_tuple(camera.coverage_polygon_world),
                    tuple(camera.overlap_camera_ids),
                )
                for camera in sorted(self.project_config.cameras, key=lambda item: item.camera_id)
            ),
            overlap.enabled,
            overlap.overlap_area_min_m2,
            overlap.boundary_gap_m,
        )

    def _heatmap_viewport_key(self) -> tuple[object, ...]:
        return (
            self._viewport_tuple(self.project_config.venue_map.manual_viewport_override),
            tuple(
                (
                    zone.zone_id,
                    self._point_tuple(zone.polygon_world),
                )
                for zone in self.project_config.venue_map.zones
            ),
            tuple(
                (
                    anchor.anchor_id,
                    self._point(anchor.world_point),
                )
                for anchor in self.project_config.shared_anchors
            ),
            tuple(
                (
                    camera.camera_id,
                    camera.calibration_valid,
                    self._point_tuple(camera.coverage_polygon_world),
                )
                for camera in sorted(self.project_config.cameras, key=lambda item: item.camera_id)
            ),
        )

    @staticmethod
    def _point_tuple(points: list[Point] | None) -> tuple[tuple[float, float], ...]:
        return tuple(MultiCameraPipelineManager._point(point) for point in points or [])

    @staticmethod
    def _point(point: Point) -> tuple[float, float]:
        return (round(point[0], 6), round(point[1], 6))

    @staticmethod
    def _viewport_tuple(viewport: WorldViewport | None) -> tuple[float, float, float, float] | None:
        if viewport is None:
            return None
        return (
            round(viewport.min_x, 6),
            round(viewport.min_y, 6),
            round(viewport.max_x, 6),
            round(viewport.max_y, 6),
        )
