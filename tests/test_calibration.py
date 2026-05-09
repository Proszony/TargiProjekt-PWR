import unittest

from core.calibration import compute_homography, project_point


class CalibrationTests(unittest.TestCase):
    def test_homography_projects_square_to_world_coordinates(self) -> None:
        matrix = compute_homography(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        )
        self.assertIsNotNone(matrix)
        point = project_point(matrix, (50.0, 50.0))
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[0], 5.0, places=2)
        self.assertAlmostEqual(point[1], 2.5, places=2)


if __name__ == "__main__":
    unittest.main()
