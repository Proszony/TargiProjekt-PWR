import unittest

try:
    from ui.camera_grid_view import CameraGridView
except ModuleNotFoundError as exc:  # pragma: no cover - optional UI dependency
    CameraGridView = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(CameraGridView is None, f"UI dependencies unavailable: {_IMPORT_ERROR}")
class CameraGridLayoutTests(unittest.TestCase):
    def test_grid_columns_for_one_camera(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(1), 1)

    def test_grid_columns_for_two_cameras(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(2), 1)

    def test_grid_columns_for_three_cameras(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(3), 2)

    def test_grid_columns_for_four_cameras(self) -> None:
        self.assertEqual(CameraGridView._grid_columns_for_count(4), 2)


if __name__ == "__main__":
    unittest.main()
