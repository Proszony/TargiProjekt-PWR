import tempfile
import unittest
from pathlib import Path

from core.model_catalog import available_detection_models


class ModelListingTests(unittest.TestCase):
    def test_dynamic_model_listing_uses_models_folder_and_filters_non_detect_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models_dir = root / "models"
            models_dir.mkdir()
            (models_dir / "yolo26m.pt").write_bytes(b"")
            (models_dir / "yolo26s.pt").write_bytes(b"")
            (models_dir / "yolo26m-seg.pt").write_bytes(b"")
            (models_dir / "yolo11n-pose.pt").write_bytes(b"")

            models = available_detection_models(root)
            paths = [path for _label, path in models]

            self.assertIn("models/yolo26m.pt", paths)
            self.assertIn("models/yolo26s.pt", paths)
            self.assertNotIn("models/yolo26m-seg.pt", paths)
            self.assertNotIn("models/yolo11n-pose.pt", paths)


if __name__ == "__main__":
    unittest.main()
