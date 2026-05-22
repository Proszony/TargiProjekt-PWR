import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QSizePolicy

    from core.models import CameraConfig
    from ui.camera_grid_view import CameraGridView
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
    CameraGridView = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(CameraGridView is None, f"UI dependencies unavailable: {_IMPORT_ERROR}")
class CameraGridLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])

    def test_grid_columns_for_one_camera(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(1), 1)

    def test_grid_columns_for_two_cameras(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(2), 1)

    def test_grid_columns_for_three_cameras(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(3), 2)

    def test_grid_columns_for_four_cameras(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(4), 2)

    def test_single_camera_panel_expands_without_empty_grid_rows(self) -> None:
        grid = CameraGridView()
        grid.set_cameras([CameraConfig(camera_id="camera-1", enabled=True)])

        self.assertEqual(len(grid._panels), 1)
        panel = grid._panels["camera-1"]
        self.assertEqual(panel.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(panel.sizePolicy().verticalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(panel.image_label.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(panel.image_label.sizePolicy().verticalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(grid.layout().rowStretch(0), 1)
        self.assertEqual(grid.layout().columnStretch(0), 1)


if __name__ == "__main__":
    unittest.main()
