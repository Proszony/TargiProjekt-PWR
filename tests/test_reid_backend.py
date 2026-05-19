import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.models import ReIDConfig
try:
    from core.reid_manager import ReIDManager
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
    ReIDManager = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(ReIDManager is None, f"ReID runtime dependencies unavailable: {_IMPORT_ERROR}")
class ReIDBackendTests(unittest.TestCase):
    def test_disabled_backend_reports_degraded_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReIDManager(ReIDConfig(enabled=False), Path(temp_dir))
            status = manager.status()
            self.assertFalse(status.available)

    def test_unsupported_optional_backend_reports_degraded_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReIDManager(
                ReIDConfig(
                    enabled=True,
                    backend="fastreid",
                    weights_path="models/reid/missing.onnx",
                    download_if_missing=False,
                ),
                Path(temp_dir),
            )
            status = manager.status()
            self.assertFalse(status.available)
            self.assertEqual(status.backend_name, "fastreid")

    def test_disabled_backend_falls_back_to_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ReIDManager(ReIDConfig(enabled=False), Path(temp_dir))
            manager.ensure_loaded()
            frame = np.zeros((256, 128, 3), dtype=np.uint8)
            frame[:, :, 1] = 127
            observations = manager._backend.embed_batch([(frame, (0, 0, 128, 256))])  # type: ignore[attr-defined]
            self.assertTrue(observations)
            self.assertGreater(len(observations[0]), 0)


if __name__ == "__main__":
    unittest.main()
