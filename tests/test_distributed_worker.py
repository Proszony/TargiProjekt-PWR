from __future__ import annotations

import unittest

try:
    from PySide6.QtGui import QColor, QImage

    from core import runtime_defaults as rd
    from core.distributed_worker import DistributedCameraWorker
    from core.models import DistributedRuntimeConfig, ProjectConfig
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


if __name__ == "__main__":
    unittest.main()
