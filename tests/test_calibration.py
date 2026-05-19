import unittest

from core.calibration import (
    compute_world_viewport,
    compute_homography,
    compute_homography_result,
    diagnose_coverage_polygon,
    project_point,
    recompute_camera_coverage,
)
from core.models import CameraConfig, VenueMapConfig, ZoneDefinition


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

    def test_homography_result_reports_rmse_and_validity(self) -> None:
        result = compute_homography_result(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)],
        )
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.reprojection_rmse_px)
        self.assertIsNotNone(result.max_reprojection_error_px)
        self.assertEqual(result.used_anchor_count, 4)

    def test_world_viewport_auto_fits_polygons(self) -> None:
        viewport = compute_world_viewport(
            [
                CameraConfig(calibration_valid=True, coverage_polygon_world=[(-1.0, 2.0), (4.0, 2.0), (4.0, 5.0)]),
                CameraConfig(calibration_valid=True, coverage_polygon_world=[(6.0, -2.0), (12.0, -2.0), (12.0, 3.0)]),
            ],
        )
        self.assertLessEqual(viewport.min_x, -1.0)
        self.assertLessEqual(viewport.min_y, -2.0)
        self.assertGreaterEqual(viewport.max_x, 12.0)
        self.assertGreaterEqual(viewport.max_y, 5.0)

    def test_bad_coverage_geometry_triggers_warnings(self) -> None:
        warnings = diagnose_coverage_polygon(
            [(-2.0, 1.0), (19.0, 1.0), (19.0, 18.0), (-2.0, 18.0)],
            [(0.0, 1.0), (9.0, 1.0), (9.0, 12.0), (0.0, 12.0)],
        )
        self.assertTrue(any("sanitization" in warning.lower() or "significantly" in warning.lower() for warning in warnings))

    def test_fewer_than_four_anchors_is_invalid(self) -> None:
        result = compute_homography_result(
            [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)],
            [(0.0, 0.0), (10.0, 0.0), (10.0, 5.0)],
        )
        self.assertFalse(result.is_valid)
        self.assertIsNone(result.homography_image_to_world)

    def test_recompute_camera_coverage_projects_from_image_polygon(self) -> None:
        camera = CameraConfig(
            homography_image_to_world=[[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 1.0]],
            coverage_polygon_image=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)],
            calibration_valid=True,
            frame_width=100,
            frame_height=100,
        )
        result = recompute_camera_coverage(camera)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.raw_polygon_world), 4)
        self.assertEqual(len(result.sanitized_polygon_world), 4)
        self.assertAlmostEqual(result.raw_polygon_world[1][0], 5.0, places=2)

    def test_recompute_camera_coverage_requires_image_polygon(self) -> None:
        camera = CameraConfig(
            homography_image_to_world=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            calibration_valid=True,
        )
        result = recompute_camera_coverage(camera)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("image-space coverage polygon" in warning for warning in result.warnings))

    def test_world_viewport_includes_zones(self) -> None:
        viewport = compute_world_viewport(
            [],
            [
                ZoneDefinition(
                    zone_id="zone-1",
                    name="Zone 1",
                    kind="neutral",
                    polygon_world=[(10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0)],
                )
            ],
        )
        self.assertLessEqual(viewport.min_x, 10.0)
        self.assertGreaterEqual(viewport.max_x, 14.0)


if __name__ == "__main__":
    unittest.main()
