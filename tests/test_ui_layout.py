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

    def test_single_camera_stage_expands_and_is_selected(self) -> None:
        grid = CameraGridView()
        grid.set_cameras([CameraConfig(camera_id="camera-1", enabled=True)])

        self.assertEqual(grid.selected_camera_id(), "camera-1")
        self.assertEqual(grid.stage.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(grid.stage.sizePolicy().verticalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(grid.stage.image_label.sizePolicy().horizontalPolicy(), QSizePolicy.Expanding)
        self.assertEqual(grid.stage.image_label.sizePolicy().verticalPolicy(), QSizePolicy.Expanding)

    def test_transition_from_two_cameras_to_one_selects_remaining_camera(self) -> None:
        grid = CameraGridView()
        grid.set_cameras(
            [
                CameraConfig(camera_id="camera-1", enabled=True, display_order=0),
                CameraConfig(camera_id="camera-2", enabled=True, display_order=1),
            ]
        )
        grid.set_selected_camera("camera-2")
        grid.set_cameras([CameraConfig(camera_id="camera-1", enabled=True, display_order=0)])

        self.assertEqual(grid.selected_camera_id(), "camera-1")
        self.assertEqual(grid.selected_camera_state().camera_id, "camera-1")


if __name__ == "__main__":
    unittest.main()
