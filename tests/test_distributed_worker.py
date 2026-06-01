from __future__ import annotations

import unittest

try:
    from PySide6.QtGui import QColor, QImage

    from core import runtime_defaults as rd
    from core.distributed_serialization import worker_config_to_network_dict
    from core.distributed_worker import DistributedCameraWorker
    from core.models import CameraConfig, DistributedRuntimeConfig, ProjectConfig, VenueMapConfig
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI/runtime dependency
    DistributedCameraWorker = None
    QImage = None
    QColor = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(DistributedCameraWorker is None, f"UI/runtime dependencies unavailable: {_IMPORT_ERROR}")
class DistributedWorkerTests(unittest.TestCase):
    def test_preview_encoding_scales_large_images_before_jpeg(self) -> None:
        worker = DistributedCameraWorker.__new__(DistributedCameraWorker)
        worker.project_config = ProjectConfig(
            distributed_runtime=DistributedRuntimeConfig(preview_jpeg_quality=80)
        )

        image = QImage(1920, 1080, QImage.Format_RGB32)
        image.fill(QColor("#111827"))

        jpeg_bytes = worker._encode_jpeg(image)
        decoded = QImage.fromData(jpeg_bytes, "JPEG")

        self.assertFalse(decoded.isNull())
        self.assertLessEqual(decoded.width(), rd.DEFAULT_DISTRIBUTED_PREVIEW_MAX_WIDTH)

    def test_missing_local_camera_config_bootstraps_placeholder(self) -> None:
        worker = DistributedCameraWorker.__new__(DistributedCameraWorker)
        worker.camera_id = "camera-9"

        config = worker._camera_config(ProjectConfig(cameras=[]))

        self.assertEqual(config.camera_id, "camera-9")
        self.assertEqual(config.runtime_mode, "remote")

    def test_worker_config_payload_updates_runtime_config(self) -> None:
        worker = DistributedCameraWorker.__new__(DistributedCameraWorker)
        worker.camera_id = "camera-1"
        worker.camera_config = CameraConfig(camera_id="camera-1", source_value="udp://0.0.0.0:5000")
        worker.project_config = ProjectConfig(
            venue_map=VenueMapConfig(map_image_path="old.png"),
            distributed_runtime=DistributedRuntimeConfig(preview_fps=1.0),
        )
        worker._playback_sync = worker.project_config.playback_sync
        worker._pipeline_worker = None
        worker._pipeline_thread = None
        worker._server_config_hash = ""

        server_project = ProjectConfig(
            venue_map=VenueMapConfig(map_image_path="server.png"),
            distributed_runtime=DistributedRuntimeConfig(preview_fps=4.0),
        )
        server_camera = CameraConfig(
            camera_id="camera-1",
            runtime_mode="remote",
            remote_worker_id="edge-1",
            source_value="udp://127.0.0.1:7001",
        )

        worker._apply_worker_config(worker_config_to_network_dict(server_project, server_camera))

        self.assertEqual(worker.camera_config.source_value, "udp://127.0.0.1:7001")
        self.assertEqual(worker.camera_config.remote_worker_id, "edge-1")
        self.assertEqual(worker.project_config.venue_map.map_image_path, "server.png")
        self.assertAlmostEqual(worker.project_config.distributed_runtime.preview_fps, 4.0)
        self.assertAlmostEqual(worker._preview_interval_s, 0.25)


if __name__ == "__main__":
    unittest.main()
