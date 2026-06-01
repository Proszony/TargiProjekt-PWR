import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QSizePolicy

    from core.models import CameraConfig
    from ui.camera_manager_dialog import CameraEditorDialog
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


@unittest.skipIf(CameraGridView is None, f"UI dependencies unavailable: {_IMPORT_ERROR}")
class CameraEditorDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._app = QApplication.instance() or QApplication([])

    def test_project_mp4_row_is_hidden_for_udp_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dialog = CameraEditorDialog(
                CameraConfig(source_type="udp", source_value="udp://0.0.0.0:5012"),
                all_camera_ids=[],
                detector_models=[("Detector", "model.pt")],
                project_root=Path(temp_dir),
            )

            self.assertTrue(dialog.project_mp4_label.isHidden())
            self.assertTrue(dialog.project_mp4_combo.isHidden())
            self.assertFalse(dialog.udp_port_label.isHidden())
            self.assertFalse(dialog.udp_port_spin.isHidden())
            self.assertEqual(dialog.udp_port_spin.value(), 5012)

    def test_udp_port_sets_source_value_on_accept(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dialog = CameraEditorDialog(
                CameraConfig(source_type="file", source_value="demo.mp4"),
                all_camera_ids=[],
                detector_models=[("Detector", "model.pt")],
                project_root=Path(temp_dir),
            )
            dialog.source_type_combo.setCurrentIndex(0)
            dialog.udp_port_spin.setValue(6024)

            dialog._accept_if_valid()

            self.assertEqual(dialog.camera_config.source_type, "udp")
            self.assertEqual(dialog.camera_config.source_value, "udp://0.0.0.0:6024")
            self.assertFalse(dialog.camera_config.loop_file)


if __name__ == "__main__":
    unittest.main()
