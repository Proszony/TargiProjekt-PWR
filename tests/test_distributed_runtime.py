from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.models import (
    AnalyticsConfig,
    CameraConfig,
    CameraTrackingPacket,
    DistributedRuntimeConfig,
    LocalTrack,
    MultiCameraRuntimeSnapshot,
    ProjectConfig,
    VenueMapConfig,
    ZoneDefinition,
)
from core.statistics_service import StatisticsService

try:
    from PySide6.QtCore import QByteArray, QBuffer, QIODevice
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    from core.multi_camera_runtime import MultiCameraPipelineManager
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
    MultiCameraPipelineManager = None
    QByteArray = None
    QBuffer = None
    QIODevice = None
    QColor = None
    QImage = None
    QApplication = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _packet(
    camera_id: str,
    person_id: str,
    point: tuple[float, float],
    appearance: list[float],
    *,
    frame_index: int = 1,
    fps: float = 0.0,
    source_fps: float | None = None,
    processing_latency_s: float = 0.0,
) -> CameraTrackingPacket:
    return CameraTrackingPacket(
        camera_id=camera_id,
        timestamp=1.0,
        frame_index=frame_index,
        source_fps=source_fps,
        processing_latency_s=processing_latency_s,
        local_tracks={
            1: LocalTrack(
                camera_id=camera_id,
                local_track_id=1,
                current_bbox_xyxy=(10, 10, 30, 60),
                ground_anchor_world=point,
                smoothed_ground_anchor_world=point,
                confidence=0.9,
                first_seen_ts=1.0,
                last_seen_ts=1.0,
                appearance_descriptor=list(appearance),
                active=True,
            )
        },
        frame_size=(640, 480),
        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
        fps=fps,
    )


def _jpeg_preview(camera_id: str, *, frame_index: int) -> dict[str, object]:
    if QImage is None or QColor is None or QByteArray is None or QBuffer is None or QIODevice is None:
        raise RuntimeError("Qt preview helpers unavailable.")
    image = QImage(640, 360, QImage.Format_RGB32)
    image.fill(QColor("#020617"))
    encoded = QByteArray()
    buffer = QBuffer(encoded)
    if not buffer.open(QIODevice.WriteOnly):
        raise RuntimeError("Failed to open preview buffer.")
    image.save(buffer, "JPEG", 80)
    buffer.close()
    return {
        "camera_id": camera_id,
        "frame_index": frame_index,
        "timestamp": 1.0,
        "width": image.width(),
        "height": image.height(),
        "jpeg_bytes": bytes(encoded),
    }


@unittest.skipIf(MultiCameraPipelineManager is None, f"UI dependencies unavailable: {_IMPORT_ERROR}")
class DistributedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if QApplication is not None:
            cls._app = QApplication.instance() or QApplication([])

    def test_single_file_camera_uses_realtime_session_mode(self) -> None:
        mode = MultiCameraPipelineManager._determine_session_sync_mode(
            [CameraConfig(camera_id="camera-1", source_type="file")]
        )

        self.assertEqual(mode, "single_file_realtime")

    def test_single_file_realtime_uses_local_playback_anchor(self) -> None:
        before = time.perf_counter()
        anchor = MultiCameraPipelineManager.worker_file_playback_started_wall_time(
            "single_file_realtime",
            session_started_at_unix_s=time.time() - 30.0,
        )
        after = time.perf_counter()

        self.assertIsNotNone(anchor)
        assert anchor is not None
        self.assertGreaterEqual(anchor, before)
        self.assertLessEqual(anchor, after)

    def test_remote_packet_ingestion_updates_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                venue_map=VenueMapConfig(
                    zones=[
                        ZoneDefinition(
                            zone_id="booth-1",
                            name="Booth 1",
                            kind="booth",
                            polygon_world=[(1.0, 1.0), (3.5, 1.0), (3.5, 3.5), (1.0, 3.5)],
                        )
                    ]
                ),
                analytics=AnalyticsConfig(zone_entry_min_duration_s=0.0),
                cameras=[
                    CameraConfig(
                        camera_id="camera-local",
                        runtime_mode="local",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    ),
                    CameraConfig(
                        camera_id="camera-remote",
                        runtime_mode="remote",
                        remote_worker_id="edge-1",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.5, 0.0), (4.5, 0.0), (4.5, 4.0), (0.5, 4.0)],
                    ),
                ],
                distributed_runtime=DistributedRuntimeConfig(enabled=True),
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            snapshots: list[MultiCameraRuntimeSnapshot] = []
            manager.runtime_snapshot_ready.connect(lambda snapshot: snapshots.append(snapshot))

            manager._handle_camera_packet(_packet("camera-local", "camera-local:P1", (2.0, 2.0), [1.0, 0.0, 0.0]))
            manager._handle_remote_packet(
                _packet("camera-remote", "camera-remote:P1", (2.1, 2.0), [0.99, 0.01, 0.0])
            )

            self.assertTrue(snapshots)
            latest = snapshots[-1]
            self.assertEqual(set(latest.camera_packets), {"camera-local", "camera-remote"})
            self.assertGreaterEqual(latest.active_map_presence_count, 1)
            self.assertEqual(latest.analytics_snapshot.total_current_occupancy, 1)

    def test_remote_packet_updates_camera_fps_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                cameras=[
                    CameraConfig(
                        camera_id="camera-remote",
                        runtime_mode="remote",
                        remote_worker_id="edge-1",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    )
                ],
                distributed_runtime=DistributedRuntimeConfig(enabled=True),
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            fps_updates: list[tuple[str, float]] = []
            manager.camera_fps_changed.connect(lambda camera_id, fps: fps_updates.append((camera_id, fps)))

            manager._handle_remote_packet(
                _packet(
                    "camera-remote",
                    "camera-remote:P1",
                    (2.0, 2.0),
                    [1.0, 0.0, 0.0],
                    fps=17.5,
                )
            )

            self.assertIn(("camera-remote", 17.5), fps_updates)

    def test_remote_live_packets_are_coalesced_between_processing_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                cameras=[
                    CameraConfig(
                        camera_id="camera-remote",
                        runtime_mode="remote",
                        remote_worker_id="edge-1",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    )
                ],
                distributed_runtime=DistributedRuntimeConfig(enabled=True),
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            calls: list[dict[str, object]] = []

            def record_process_group(**kwargs: object) -> None:
                calls.append(kwargs)

            manager._process_packet_group = record_process_group  # type: ignore[method-assign]
            manager._last_remote_live_processed_at = time.monotonic()

            first = _packet(
                "camera-remote",
                "camera-remote:P1",
                (2.0, 2.0),
                [1.0, 0.0, 0.0],
                frame_index=1,
            )
            second = _packet(
                "camera-remote",
                "camera-remote:P1",
                (2.1, 2.0),
                [1.0, 0.0, 0.0],
                frame_index=2,
            )
            second.timestamp = first.timestamp + 0.01

            manager._handle_remote_packet(first)
            manager._handle_remote_packet(second)

            self.assertEqual(calls, [])
            self.assertTrue(manager._remote_live_process_timer.isActive())
            manager._remote_live_process_timer.stop()
            manager._process_pending_remote_live_packets()

            self.assertEqual(len(calls), 1)
            camera_packets = calls[0]["camera_packets"]
            self.assertIsInstance(camera_packets, dict)
            self.assertEqual(camera_packets["camera-remote"].frame_index, 2)

    def test_local_operator_preview_uses_worker_gated_frames_without_skipping_analytics(self) -> None:
        if QImage is None or QColor is None:
            self.skipTest("Qt image helpers unavailable.")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                cameras=[
                    CameraConfig(
                        camera_id="camera-local",
                        runtime_mode="local",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    )
                ],
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            rendered_frames: list[QImage] = []
            snapshots: list[MultiCameraRuntimeSnapshot] = []
            manager.camera_frame_ready.connect(
                lambda camera_id, image: rendered_frames.append(image)
                if camera_id == "camera-local" and isinstance(image, QImage)
                else None
            )
            manager.runtime_snapshot_ready.connect(lambda snapshot: snapshots.append(snapshot))
            frame = QImage(320, 180, QImage.Format_RGB32)
            frame.fill(QColor("#020617"))

            first_packet = _packet(
                "camera-local",
                "camera-local:P1",
                (2.0, 2.0),
                [1.0, 0.0, 0.0],
                frame_index=1,
            )
            second_packet = _packet(
                "camera-local",
                "camera-local:P1",
                (2.0, 2.0),
                [1.0, 0.0, 0.0],
                frame_index=2,
            )
            first_packet.local_tracks[1].current_bbox_xyxy = None
            first_packet.local_tracks[1].last_bbox_xyxy_for_matching = None
            second_packet.local_tracks[1].current_bbox_xyxy = None
            second_packet.local_tracks[1].last_bbox_xyxy_for_matching = None
            manager._handle_camera_frame("camera-local", 1, frame)
            manager._handle_camera_packet(first_packet)
            manager._handle_camera_packet(second_packet)

            self.assertEqual(len(rendered_frames), 1)
            self.assertEqual(len(snapshots), 2)

    def test_remote_preview_refreshes_when_labels_change_without_packet_frame_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                cameras=[
                    CameraConfig(
                        camera_id="camera-remote",
                        runtime_mode="remote",
                        remote_worker_id="edge-1",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    )
                ],
                distributed_runtime=DistributedRuntimeConfig(enabled=True),
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            rendered_frames: list[QImage] = []
            manager.camera_frame_ready.connect(
                lambda camera_id, image: rendered_frames.append(image)
                if camera_id == "camera-remote" and isinstance(image, QImage)
                else None
            )

            manager._handle_remote_preview_frame(_jpeg_preview("camera-remote", frame_index=10))
            manager._handle_remote_packet(
                _packet(
                    "camera-remote",
                    "camera-remote:P1",
                    (2.0, 2.0),
                    [1.0, 0.0, 0.0],
                    frame_index=12,
                )
            )

            self.assertEqual(len(rendered_frames), 2)

    def test_remote_preview_skips_relabel_when_preview_and_labels_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project = ProjectConfig(
                cameras=[
                    CameraConfig(
                        camera_id="camera-remote",
                        runtime_mode="remote",
                        remote_worker_id="edge-1",
                        calibration_valid=True,
                        coverage_polygon_world=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)],
                    )
                ],
                distributed_runtime=DistributedRuntimeConfig(enabled=True),
            )
            manager = MultiCameraPipelineManager(project, StatisticsService(root), root)
            rendered_frames: list[QImage] = []
            manager.camera_frame_ready.connect(
                lambda camera_id, image: rendered_frames.append(image)
                if camera_id == "camera-remote" and isinstance(image, QImage)
                else None
            )

            manager._handle_remote_preview_frame(_jpeg_preview("camera-remote", frame_index=10))
            packet = _packet(
                "camera-remote",
                "camera-remote:P1",
                (2.0, 2.0),
                [1.0, 0.0, 0.0],
                frame_index=12,
            )
            manager._handle_remote_packet(packet)
            manager._handle_remote_packet(packet)

            self.assertEqual(len(rendered_frames), 2)


if __name__ == "__main__":
    unittest.main()
